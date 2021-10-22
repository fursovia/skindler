from pathlib import Path
import json
from tqdm import tqdm

import torch
import torch.nn.functional
from transformers import MarianMTModel, MarianTokenizer
from transformers.trainer_utils import get_last_checkpoint
from typer import Typer

from skindler import MODEL_NAME, MAX_LENGTH
from skindler.models import MarianAutoEncoder, Bleuer


MODELS_FOLDER = (Path(__file__).parent / ".." / "..").resolve() / "models"


app = Typer()


def attack(
        text: str,
        tokenizer: MarianTokenizer,
        autoencoder: MarianAutoEncoder,
        bleuer: Bleuer,
        epsilon: float = 0.25,
        device: torch.device = torch.device('cuda')
):

    for params in bleuer.parameters():
        params.grad = None

    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    embeddings = autoencoder.get_embeddings(**inputs)
    embeddings.requires_grad = True

    bleu = bleuer.get_logits(embeddings)
    loss = torch.nn.functional.l1_loss(bleu, torch.tensor(1.0, device=device))
    loss.backward()

    embeddings_grad = embeddings.grad.data
    perturbed_embeddings = embeddings + epsilon * embeddings_grad.sign()

    logits = autoencoder.forward_on_embeddings(perturbed_embeddings)
    ids = logits.argmax(dim=-1)
    decoded = tokenizer.decode(ids[0].cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded


@app.command()
def main(
        dataset_path: Path,
        ae_dir: Path = Path('experiments/ae'),
        bl_dir: Path = Path('experiments/bleuer'),
        save_to: Path = Path('results.json'),
        epsilon: float = 0.25,
):
    device = torch.device("cuda")

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).eval().to(device)
    ae_path = str(Path(get_last_checkpoint(str(ae_dir))) / 'pytorch_model.bin')
    bl_path = str(Path(get_last_checkpoint(str(bl_dir))) / 'pytorch_model.bin')
    autoencoder = MarianAutoEncoder(MODEL_NAME)
    autoencoder.load_state_dict(torch.load(ae_path))
    autoencoder.to(device)

    bleuer = Bleuer(MODEL_NAME)
    bleuer.load_state_dict(torch.load(bl_path))
    bleuer.to(device)

    data = []
    with dataset_path.open() as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))

    with save_to.open('w') as f:
        for example in tqdm(data):
            en = example['en']
            ru = example['ru']
            ru_trans = example['ru_trans']
            # bleu = example['bleu']

            en_attacked = attack(
                en, tokenizer=tokenizer, autoencoder=autoencoder, bleuer=bleuer, epsilon=epsilon, device=device
            )

            batch_text_inputs = tokenizer(
                en_attacked,
                max_length=MAX_LENGTH,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            batch_text_inputs.to(device)

            output = model.generate(**batch_text_inputs)
            translations = tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            f.write(
                json.dumps(
                    {
                        'en': en,
                        'ru': ru,
                        'ru_trans': ru_trans,
                        'en_attacked': en_attacked,
                        'ru_trans_attacked': translations
                    },
                    ensure_ascii=False
                ) + '\n'
            )


if __name__ == "__main__":
    app()
