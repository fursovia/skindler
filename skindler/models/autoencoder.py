from typing import List
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import typer
from nltk.translate.bleu_score import sentence_bleu

from skindler import MODEL_NAME, MAX_LENGTH
from skindler.dataset import SkDataset


app = typer.Typer()


class AutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.encoder = MarianMTModel.from_pretrained(MODEL_NAME).get_encoder().eval()
        self.linear = torch.nn.Linear(512, self.tokenizer.vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

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

        logits = self.linear(embeddings)
        return logits, inputs["input_ids"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, input_ids = self(x)
        loss = self.loss(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, input_ids = self(x)
        loss = self.loss(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), )
        new_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        bleus = []
        for orig, dec in zip(x, decoded):
            bleus.append(sentence_bleu([dec.lower()], orig.lower()))

        self.log('bleu', sum(bleus) / len(bleus))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@app.command()
def train(
        train_path: Path = typer.Option(..., exists=True, dir_okay=False),
        valid_path: Path = typer.Option(..., exists=True, dir_okay=False),
        save_to: Path = typer.Option(..., file_okay=False),
        batch_size: int = 128
):
    train_dataset = SkDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SkDataset(valid_path)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model = AutoEncoder()

    trainer = pl.Trainer(gpus=1, default_root_dir=str(save_to))
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    app()
