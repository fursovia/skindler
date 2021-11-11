import json
import math
from pathlib import Path

import torch
import typer
from datasets import concatenate_datasets
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast

from skindler import DATASET_NAME, MODEL_NAME, MAX_LENGTH, MBART_NAME, SRC_LNG, TGT_LNG
from skindler.modules.metrics import ALL_METRICS
from skindler.utils import lazy_groups_of

app = typer.Typer()

COLLECT_DATASET_CONFIG = {
    'bleuer': {
        'model': MODEL_NAME,
        'target_metric_name': 'bleu',
        'target_metric_function': sentence_bleu,
        'model_class': MarianMTModel,
        'tokenizer_class': MarianTokenizer},
    'mbart_bertscore': {
        'model': MBART_NAME,
        'target_metric_name': 'bertscore',
        'target_metric_function': ALL_METRICS['bertscore'],
        'model_class': MBartForConditionalGeneration,
        'tokenizer_class': MBart50TokenizerFast}}


@app.command()
def get_dataset(
        task: str,
        save_to: Path,
        split: str = 'train',
        batch_size: int = 128,
        device: int = -1,
        samples_to_use: int = 10000):
    assert task in COLLECT_DATASET_CONFIG.keys()
    config = COLLECT_DATASET_CONFIG[task]

    dataset = load_dataset(*DATASET_NAME)
    splits_to_use = [split]
    dataset = concatenate_datasets([dataset[split] for split in splits_to_use]).select(list(
        filter(lambda i: i < len(dataset[splits_to_use[0]]),
               torch.arange(samples_to_use).tolist())))

    typer.echo(f"Dataset len : {len(dataset)}; split: {split}")

    tokenizer = config['tokenizer_class'].from_pretrained(config['model'])
    model = config['model_class'].from_pretrained(config['model'])
    device = torch.device(f'cuda:{device}') if device != -1 else -1
    model = model.to(device)
    model = model.eval()
    typer.echo(f"Loaded model")

    all_x = []
    all_y = []
    all_translations = []

    for batch in tqdm(lazy_groups_of(dataset, batch_size),
                      total=math.ceil(len(dataset) / batch_size), miniters=20):
        x = [example['translation'][SRC_LNG] for example in batch]
        y = [example['translation'][TGT_LNG] for example in batch]

        batch_text_inputs = tokenizer(
            x,
            max_length=MAX_LENGTH,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        batch_text_inputs.to(device)

        if task == 'mbart_bertscore':
            output = model.generate(
                **batch_text_inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"])
        else:
            output = model.generate(**batch_text_inputs)
        # TODO: remove special tokens (__)
        translations = tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        all_x.extend(x)
        all_y.extend(y)
        all_translations.extend(translations)

    typer.echo(f"Counting target metric")
    all_target_metric = config['target_metric_function'](
        [i.lower() for i in all_translations], [i.lower() for i in all_y])

    typer.echo(f"Saving result dataset to {str(save_to)}")
    with save_to.open("w") as f:
        for x_sentence, y_sentence, y_translated_sentence, target_metric in zip(
                all_x, all_y, all_translations, all_target_metric):
            f.write(
                json.dumps(
                    {
                        "x": x_sentence,
                        "y": y_sentence,
                        "y_trans": y_translated_sentence,
                        config['target_metric_name']: target_metric
                    },
                    ensure_ascii=False
                ) + "\n"
            )


if __name__ == '__main__':
    app()
