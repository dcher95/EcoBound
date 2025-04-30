import pytorch_lightning as pl
import torch
import torch.nn as nn
# from transformers import Dinov2Model
from losses import get_losses
from utils import load_species_weights
from networks import ResidualFCNet

class SDM(pl.LightningModule):
    def __init__(self, 
                 species_file = "./data/species.npy", 
                 species_counts_file = "./data/species_counts.npy",
                 loss_type='an_full', 
                 pos_weight = 1,
                 species_weights_method="uniform",
                 learning_rate=1e-4):
        super(SDM, self).__init__()

        # TODO: Add option to include satellite imagery.
        # TODO: Add option to do different weighting of target & background
        # TODO: Add option to include LANDSCAN (population)

        self.save_hyperparameters()

        # Load species data and precompute species weights 
        species_to_index, species_weights, num_classes = load_species_weights(
            species_file, species_counts_file, species_weights_method
        )
        self.num_classes = num_classes

        # Register species weights as a persistent buffer.
        self.register_buffer('species_weights', species_weights)

        self.loc_encoder = ResidualFCNet()
        self.class_projector = nn.Linear(256, self.num_classes, bias=False)

        self.pos_weight = (self.num_classes - 1) if pos_weight == 'num_classes' else pos_weight

        # Get the loss function based on loss_type
        self.loss_type = loss_type
        self.learning_rate = learning_rate

        # Let utils handle loss-specific kwargs
        self.loss_fn = get_losses(
            loss_type=self.loss_type,
            pos_weight=self.pos_weight,
            species_weights=self.species_weights 
        )

    def forward(self, loc_feats):
        x = self.loc_encoder(loc_feats)
        pred = self.class_projector(x)
        return pred
    
    def forward_location_features(self, loc_feats):
        return self.loc_encoder(loc_feats)
    
    def forward_species(self,loc_feats, class_of_interest=None):
        x = self.loc_encoder(loc_feats)
        class_pred = self.class_projector(x)
        if class_of_interest is not None:
            class_pred = class_pred[:, class_of_interest]
        return class_pred
    
    def shared_step(self, batch):
        target, feats, rand_feats = batch
        
        logits = self(feats).sigmoid()
        rand_logits = self(rand_feats).sigmoid()

        # Compute the loss and its components using the loss function.
        total_loss, components = self.loss_fn(logits, rand_logits, target)
        
        # Organize metrics for logging.
        metrics = {
            'total': total_loss,
            'target_component': components.get('target', torch.tensor(0.)),
            'non_target_component': components.get('non_target', torch.tensor(0.)),
            'background_component': components.get('background', torch.tensor(0.))
        }
        return metrics

    def _log_components(self, stage):
        # Unified logging for both loss types
        suffix = '_loss' if 'an_full' in self.loss_type else '_entropy'
        
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)