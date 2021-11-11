from pathlib import Path
from typing import List
import json
from tqdm import tqdm

import torch
import torch.nn.functional
from transformers import MarianMTModel, MarianTokenizer
from transformers.trainer_utils import get_last_checkpoint
from typer import Typer
from nltk.translate.bleu_score import sentence_bleu

from skindler import MODEL_NAME, MAX_LENGTH
from skindler.models import MarianAutoEncoder, Bleuer


MODELS_FOLDER = (Path(__file__).parent / ".." / "..").resolve() / "models"


app = Typer()


def calculate_metric(source: str, source_attacked: str, translation: str, translation_attacked: str) -> float:
    # should be large!
    source_bleu = sentence_bleu([source_attacked.lower()], source.lower())
    # should be small!
    target_bleu = sentence_bleu([translation_attacked.lower()], translation.lower())
    target_bleu_inversed = 1.0 - target_bleu

    if source_bleu or target_bleu_inversed:
        metric = 2 * (source_bleu * target_bleu_inversed) / (source_bleu + target_bleu_inversed)
    else:
        metric = 0.0
    return metric


def attack(
        text: str,
        tokenizer: MarianTokenizer,
        autoencoder: MarianAutoEncoder,
        bleuer: Bleuer,
        epsilon: float = 0.25,
        num_steps: int = 10,
        sign_mode: bool = True,
        device: torch.device = torch.device('cuda')
) -> List[str]:

    for params in bleuer.parameters():
        params.grad = None

    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    # shape [1, num_tokens, 512]
    embeddings = autoencoder.get_embeddings(**inputs)

    attacked_sentences = []
    for step in range(num_steps):
        embeddings.grad = None
        embeddings.requires_grad = True

        # shape [1, 1] [0.2 L1 loss on validation set]
        bleu = bleuer.get_logits(embeddings)
        loss = torch.nn.functional.l1_loss(bleu, torch.tensor(1.0, device=device))
        loss.backward()

        if sign_mode:
            embeddings = embeddings + epsilon * embeddings.grad.data.sign()
        else:
            embeddings = embeddings + epsilon * embeddings.grad.data
        # shape [1, num_tokens, vocab_size] [~0.02 cross entropy loss]
        logits = autoencoder.get_logits(embeddings)
        # shape [1, num_tokens]
        ids = logits.argmax(dim=-1)
        decoded = tokenizer.decode(ids[0].cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded = decoded.replace('▁', ' ').strip()
        attacked_sentences.append(decoded)
    return attacked_sentences


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
    ae_dir = get_last_checkpoint(str(ae_dir)) or ae_dir
    bl_dir = get_last_checkpoint(str(bl_dir)) or bl_dir
    ae_path = str(Path(ae_dir) / 'pytorch_model.bin')
    bl_path = str(Path(bl_dir) / 'pytorch_model.bin')
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

            en_attacked_list = attack(
                en, tokenizer=tokenizer, autoencoder=autoencoder, bleuer=bleuer, epsilon=epsilon, device=device
            )

            best_metric = 0.0
            for en_attacked in en_attacked_list:
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

                metric = calculate_metric(
                    source=en, source_attacked=en_attacked, translation=ru_trans, translation_attacked=translations
                )
                if metric > best_metric:
                    best_metric = metric
                    best_source_attacked = en_attacked
                    best_target_attacked = translations

            f.write(
                json.dumps(
                    {
                        'en': en,
                        'ru': ru,
                        'ru_trans': ru_trans,
                        'en_attacked': best_source_attacked,
                        'ru_trans_attacked': best_target_attacked
                    },
                    ensure_ascii=False
                ) + '\n'
            )


if __name__ == "__main__":
    app()
