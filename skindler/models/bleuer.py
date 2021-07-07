from typing import List

import torch
from transformers import MarianMTModel, MarianTokenizer
import pytorch_lightning as pl
import typer

from skindler import MODEL_NAME, MAX_LENGTH

app = typer.Typer()


class Bleuer(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.encoder = MarianMTModel.from_pretrained(MODEL_NAME).get_encoder().eval()
        self.linear1 = torch.nn.Linear(512 * 2, 256)
        self.linear2 = torch.nn.Linear(256, 1)
        self.loss = torch.nn.L1Loss()

    def forward(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            max_length=MAX_LENGTH,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        inputs.to(self.device)

        with torch.no_grad():
            embeddings = self.encoder(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = embeddings * mask

        embeddings = torch.cat(
            (
                torch.sum(embeddings, dim=1),
                torch.max(embeddings, dim=1).values,
            ),
            dim=1
        )
        out = self.linear1(embeddings)
        out = torch.relu(out)
        out = self.linear2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss(z.view(-1), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.loss(z.view(-1), y.view(-1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
