from pathlib import Path
import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import typer
from enum import Enum

from skindler.dataset import SkDataset
from skindler.models.autoencoder import AutoEncoder
from skindler.models.bleuer import Bleuer


app = typer.Typer()


class ModelName(str, Enum):
    BLEUER = 'bleuer'
    ENCODER = 'autoencoder'


MODELS = {
    ModelName.BLEUER: Bleuer,
    ModelName.ENCODER: AutoEncoder,
}


@app.command()
def train(
        model_name: ModelName,
        train_path: Path = typer.Option(..., exists=True, dir_okay=False),
        valid_path: Path = typer.Option(..., exists=True, dir_okay=False),
        save_to: Path = typer.Option(..., file_okay=False),
        batch_size: int = 128,
        patience: int = 5,
):
    train_dataset = SkDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SkDataset(valid_path)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model = MODELS[model_name]()

    date = datetime.datetime.utcnow().strftime('%H%M%S-%d%m')
    save_to = save_to / f"{date}_{str(model_name)}"
    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger(str(save_to / "logs")),
        gpus=1,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', dirpath=str(save_to), save_top_k=3, mode='min'),
            EarlyStopping(monitor='val_loss', patience=patience),
        ]
    )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    app()
