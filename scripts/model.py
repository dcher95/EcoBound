import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import Dinov2Model
from torch.utils.data import DataLoader
from collections import OrderedDict
from dataset import LocationDataset
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def neg_log(x):
    return -torch.log(x + 1e-5)

def bernoulli_entropy(p):
    return p * neg_log(p) + (1 - p) * neg_log(p)

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module):

    def __init__(self, num_inputs=4, num_classes=1024, num_filts=256, depth=4):
        super(ResidualFCNet, self).__init__()
        self.inc_bias = False
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        return class_pred

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]


class SDM(pl.LightningModule):
    def __init__(self, loss_type='an_full'):
        super(SDM, self).__init__()

        # TODO: Add option to include satellite imagery.
        # TODO: Add option to include inverse species weights
        # TODO: Add option to do different weighting of target & background
        # TODO: Add option to include LANDSCAN (population)

        self.save_hyperparameters()

        species_data = np.load("data/species.npy", allow_pickle=True)
        self.num_classes = len(species_data)
        
        self.loc_encoder = ResidualFCNet(
            num_inputs=4,
            num_classes=self.num_classes,
            num_filts=256,
            depth=4
        )

        self.loss_type = loss_type

    def forward(self, loc_feats):
        return self.loc_encoder(loc_feats)
    
    def shared_step(self, batch):
        y, feats, rand_feats = batch
        batch_size = y.shape[0]
        inds = torch.arange(y.shape[0])
        
        logits = self(feats).sigmoid()
        rand_logits = self(rand_feats).sigmoid()

        if self.loss_type == 'an_full':
            loss_pos = neg_log(1.0 - logits)
            loss_pos[inds[:batch_size], y.squeeze(-1)] = 1024 * neg_log(logits[inds[:batch_size], y.squeeze(-1)])
            loss_rand = neg_log(1 - rand_logits)

            return loss_pos.mean() + loss_rand.mean()
        
        elif self.loss_type == 'max_entropy':
            loss_pos = -1 * bernoulli_entropy(1.0 - logits)
            loss_pos[inds[:batch_size], y.squeeze(-1)] = 1024 * bernoulli_entropy(logits[inds[:batch_size], y.squeeze(-1)])
            loss_rand = -1 * bernoulli_entropy(1 - rand_logits)

            return loss_pos.mean() + loss_rand.mean()

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(
            LocationDataset("./data/gbif_full_filtered-train.csv"), 
            batch_size=128, 
            num_workers=16, 
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            LocationDataset("./data/gbif_full_filtered-validation.csv"), 
            batch_size=128, 
            num_workers=16, 
            shuffle=False,
            persistent_workers=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__=='__main__':
    experiment_name = 'STL-loc-base'

    model = SDM()
    wandb_logger = WandbLogger(project='ecobound', name=experiment_name)

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='{experiment_name}-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=5, 
                         logger=wandb_logger, 
                         devices=1, 
                         accelerator='gpu',
                         callbacks=[checkpoint],
                         strategy='ddp',
                         val_check_interval=0.25)
    
    trainer.fit(model)
    trainer.save_checkpoint(f"models/{experiment_name}.ckpt")
    wandb_logger.finish()