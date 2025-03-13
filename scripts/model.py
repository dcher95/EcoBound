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
    return p * neg_log(p) + (1-p) * neg_log(1-p)

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
    def __init__(self, species_file = "./data/species.npy", loss_type='an_full', target_weight = None):
        super(SDM, self).__init__()

        # TODO: Add option to include satellite imagery.
        # TODO: Add option to include inverse species weights
        # TODO: Add option to do different weighting of target & background
        # TODO: Add option to include LANDSCAN (population)

        self.save_hyperparameters()

        species_data = np.load(species_file, allow_pickle=True)
        self.num_classes = len(species_data)
        
        self.loc_encoder = ResidualFCNet(
            num_inputs=4,
            num_classes=self.num_classes,
            num_filts=256,
            depth=4
        )

        self.loss_type = loss_type

        self.target_weight = (self.num_classes - 1) if target_weight == None else target_weight

    def forward(self, loc_feats):
        return self.loc_encoder(loc_feats)
    
    def forward_location_features(self, loc_feats):
        return self.loc_encoder(loc_feats)
    
    def forward_species(self, loc_feats, class_of_interest=None):
        class_pred = self.loc_encoder(loc_feats)
        if class_of_interest is not None:
            class_pred = class_pred[:, class_of_interest]
        return class_pred
    
    def shared_step(self, batch):
        y, feats, rand_feats = batch
        batch_size = y.shape[0]
        inds = torch.arange(y.shape[0])
        
        logits = self(feats).sigmoid()
        rand_logits = self(rand_feats).sigmoid()

        # Create target mask using one-hot encoding
        target_mask = torch.zeros_like(logits, dtype=torch.bool, device=self.device)
        target_mask.scatter_(1, y, True)  # y must be shape [B, 1]

        metrics = {'total': None}
        components = {}

        if self.loss_type == 'an_full':
            loss_pos = neg_log(1.0 - logits)
            loss_pos[inds[:batch_size], y.squeeze(-1)] = self.target_weight * neg_log(logits[inds[:batch_size], y.squeeze(-1)]) # base = 1024
            loss_rand = neg_log(1.0 - rand_logits)

            components['target'] = loss_pos[target_mask].mean()
            components['non_target'] = loss_pos[~target_mask].mean()
            components['background'] = loss_rand.mean()
            metrics['total'] = loss_pos.mean() + loss_rand.mean()

        
        elif self.loss_type == 'max_entropy':
            loss_pos = -1 * bernoulli_entropy(1.0 - logits)
            loss_pos[inds[:batch_size], y.squeeze(-1)] = self.target_weight * bernoulli_entropy(logits[inds[:batch_size], y.squeeze(-1)])
            loss_rand = -1 * bernoulli_entropy(1.0 - rand_logits)

            # Calculate metrics (convert to actual entropy values)
            components['target'] = loss_pos[target_mask].mean()  # Direct entropy
            components['non_target'] = -loss_pos[~target_mask].mean()  # Remove negative
            components['background'] = -loss_rand.mean()  # Remove negative
            metrics['total'] = loss_pos.mean() + loss_rand.mean()

        # Store components with type-agnostic names
        metrics.update({
            'target_component': components['target'],
            'non_target_component': components['non_target'],
            'background_component': components['background']
        })
        
        return metrics

    def _log_components(self, stage):
        # Unified logging for both loss types
        suffix = '_loss' if self.loss_type == 'an_full' else '_entropy'
        
        self.log(f"{stage}/target{suffix}", self.metrics[f'target_component'], 
                prog_bar=(stage == 'train'))
        self.log(f"{stage}/non_target{suffix}", self.metrics['non_target_component'])
        self.log(f"{stage}/background{suffix}", self.metrics['background_component'])

    def training_step(self, batch, batch_idx):
        self.metrics = self.shared_step(batch)
        self.log("train/loss", self.metrics['total'], sync_dist=True, prog_bar=True)
        self._log_components('train')
        return self.metrics['total']

    def validation_step(self, batch, batch_idx):
        self.metrics = self.shared_step(batch)
        self.log("val/loss", self.metrics['total'], sync_dist=True, prog_bar=True)
        self._log_components('val')
        return self.metrics['total']

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__=='__main__':
    experiment_name = 'STL-loc-an_full-1024' # 'STL-loc-an_full-1024-alldata'
    loss_type = 'an_full'

    # Create datasets
    train_dataset = LocationDataset("./data/gbif_full_filtered-train.csv")
    val_dataset = LocationDataset("./data/gbif_full_filtered-validation.csv")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=16,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=16,
        shuffle=False,
        persistent_workers=False
    )

    model = SDM(loss_type = loss_type, target_weight = 1024)
    wandb_logger = WandbLogger(project='ecobound', name=experiment_name)

    checkpoint = ModelCheckpoint(
        monitor='val/loss',
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
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f"models/{experiment_name}.ckpt")