from typing import List, Optional

import torch
from transformers import MarianMTModel, MarianTokenizer
import pytorch_lightning as pl
import typer
from nltk.translate.bleu_score import sentence_bleu

from skindler import MODEL_NAME, MAX_LENGTH


app = typer.Typer()


class AutoEncoder(pl.LightningModule):

    def __init__(self, sigma: Optional[float] = None):
        super().__init__()
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.encoder = MarianMTModel.from_pretrained(MODEL_NAME).get_encoder().eval()
        self.linear = torch.nn.Linear(512, self.tokenizer.vocab_size)
        self.loss = torch.nn.CrossEntropyLoss()  # ignore_index=self.tokenizer.pad_token_id)
        self.sigma = sigma

    def forward_on_embeddings(self, embeddings: torch.Tensor):
        logits = self.linear(embeddings)
        return logits

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
            # batch_size, seq_length, 512
            embeddings = self.encoder(**inputs).last_hidden_state

        # TODO: add noisy training embeddings = embeddings + torch.rand()
        # batch_size, seq_lengyh, vocab_size
        logits = self.forward_on_embeddings(embeddings)
        return logits, inputs["input_ids"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, input_ids = self(x)
        loss = self.loss(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), )
        self.log('train_loss', loss)
        return loss

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        new_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, input_ids = self(x)
        loss = self.loss(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), )
        decoded = self.decode_logits(logits)
        bleus = []
        for orig, dec in zip(x, decoded):
            bleus.append(sentence_bleu([dec.lower().replace('▁', ' ').strip()], orig.lower()))

        self.log('bleu', sum(bleus) / len(bleus))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
