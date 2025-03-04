import os
import sys
import yaml
sys.path.insert(1, os.path.abspath(".."))
codes_dir = os.path.join('../EyeTrackDiagError/', 'lib') 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from lib import utils, models, dataloader
import torch
import torch.utils.tensorboard

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device type: {DEVICE}")

def run(config):

    torch.manual_seed(config["seed"])
    pl.seed_everything(config["seed"])
 
    ModelClass = getattr(models, config["model_class"])
    model = ModelClass(**config["model_params"]).to(DEVICE)
    model_prefix =  model.__class__.__name__

    train_loader, val_loader, test_loader = dataloader.get_train_val_test(**config)
    save_model_path = utils.create_save_folder(model_prefix, **config)

    checkpoint_callback = ModelCheckpoint(
                        monitor='val_loss',
                        dirpath= save_model_path,
                        filename='best-model',
                        save_top_k=1,
                        mode='min',
                        save_last=True
                    )

    logger = TensorBoardLogger(save_model_path)
    trainer = Trainer(max_epochs=config['num_epochs'],
                    logger=logger, 
                    enable_progress_bar=False,
                    accelerator=DEVICE.type,
                    callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    print('last model:')
    weights = torch.load(os.path.join(save_model_path, 'latest-model-v1.ckpt'))
    utils.model_eval(model, weights, test_loader, DEVICE)

    print('best model:')
    weights = torch.load(os.path.join(save_model_path, 'best-model.ckpt'))
    utils.model_eval(model, weights, test_loader, DEVICE)


if __name__ == "__main__":
    args = utils.parse_args()
    config = utils.load_config(args.config_path)

    print("Loaded configuration:\n")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    run(config)
