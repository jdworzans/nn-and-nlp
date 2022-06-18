from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from utils import PrusModule

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="model_checkpoints/",
    filename="model-{epoch:02d}-{train_loss:.2f}",
    save_top_k=-1,
    mode="min",
)

cli = LightningCLI(
    PrusModule,
    trainer_defaults={'gpus': 1, 'callbacks': [checkpoint_callback]},
    seed_everything_default=1234,
    save_config_overwrite=True
)