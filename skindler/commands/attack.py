from pathlib import Path
import json

import torch
import torch.nn.functional
from transformers import MarianMTModel, MarianTokenizer
from typer import Typer

from skindler import MODEL_NAME, MAX_LENGTH
from skindler.models import AutoEncoder, Bleuer


MODELS_FOLDER = (Path(__file__).parent / ".." / "..").resolve() / "models"


app = Typer()


@app.command()
def attack(text: str, autoencoder: AutoEncoder, bleuer: Bleuer, epsilon: float = 0.25):
    inputs = bleuer.tokenizer(
        text,
        max_length=MAX_LENGTH,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        embeddings = bleuer.encoder(**inputs).last_hidden_state

    embeddings.requires_grad = True
    bleu = bleuer.forward_on_embeddings(embeddings, inputs["attention_mask"].unsqueeze(-1))
    loss = torch.nn.functional.l1_loss(bleu, torch.tensor(1.0))
    loss.backward()

    embeddings_grad = embeddings.grad.data
    perturbed_embeddings = embeddings + epsilon * embeddings_grad.sign()

    logits = autoencoder.forward_on_embeddings(perturbed_embeddings)
    decoded = autoencoder.decode_logits(logits)
    return decoded


def main(dataset_path: Path, save_to: Path):
    device = torch.device("cuda")

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).eval().to(device)
    autoencoder = AutoEncoder.load_from_checkpoint(str(MODELS_FOLDER / "encoder.ckpt"), map_location=device)
    bleuer = Bleuer.load_from_checkpoint(str(MODELS_FOLDER / "bleuer.ckpt"), map_location=device)

    data = []
    with dataset_path.open() as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))

    with save_to.open('w') as f:
        for example in data:
            en = example['en']
            ru = example['ru']
            ru_trans = example['ru_trans']
            # bleu = example['bleu']

            en_attacked = attack(en, autoencoder=autoencoder, bleuer=bleuer)

            batch_text_inputs = tokenizer(
                en_attacked,
                max_length=MAX_LENGTH,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            batch_text_inputs.to(device)

            output = model.generate(**batch_text_inputs)
            translations = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            f.write(
                json.dumps(
                    {
                        'en': en,
                        'ru': ru,
                        'ru_trans': ru_trans,
                        'en_attacked': en_attacked,
                        'ru_trans_attacked': translations
                    }
                ) + '\n'
            )


if __name__ == "__main__":
    app()