import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from model import SDM  
from dataset import LocationDataset 
from config import config  

def main():
    experiment_name = config.experiment_name
    loss_type = config.loss_type
    pos_weight = config.pos_weight
    batch_size = config.batch_size
    train_path = config.train_path 
    val_path = config.val_path
    max_epochs = config.max_epochs
    species_weights_method = config.species_weights_method

    pl.seed_everything(config.seed, workers=True)

    # Create datasets and dataloaders.
    train_dataset = LocationDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    val_loader = None
    if val_path:
        val_dataset = LocationDataset(val_path)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, persistent_workers=False)
    
    # Initialize the model.
    model = SDM(loss_type=loss_type, 
                pos_weight=pos_weight,
                species_weights_method=species_weights_method)

    # Setup logging.
    wandb_logger = WandbLogger(project='ecobound', name=experiment_name)

    # Setup model checkpointing.
    if val_loader:
        checkpoint = ModelCheckpoint(
            dirpath='checkpoints',
            filename=f'{experiment_name}-{{epoch:02d}}-{{val/loss:.2f}}',
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            every_n_epochs=1,
            auto_insert_metric_name=False
        )
    else:
        checkpoint = ModelCheckpoint(
            dirpath='checkpoints',
            filename=f'{experiment_name}-{{epoch:02d}}',
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=1
        )
    callbacks = [checkpoint]

    # Initialize trainer.
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         logger=wandb_logger, 
                         devices=1, 
                         accelerator='gpu',
                         callbacks=callbacks,
                         strategy='ddp',
                         val_check_interval=0.25)
    
    # Start training.
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)

    # Save the final model checkpoint.
    trainer.save_checkpoint(f"./models/{experiment_name}.ckpt")

if __name__ == '__main__':
    main()
