import json
from argparse import ArgumentParser
from pathlib import Path
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from gradient_attack import GradientGuidedSearchStrategy
from metrics import ALL_METRICS
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from skindler import MODEL_NAME, MAX_LENGTH, DATASET_NAME, MAX_LENGTH, SENTENCES_TO_ATTACK



def prepare_dataloader(tokenizer):
    raw_datasets = load_dataset(*DATASET_NAME)
    del raw_datasets['train']

    source_lang = 'en'
    target_lang = 'ru'
    column_names = raw_datasets["validation"].column_names
    prefix = ""
    padding = False
    max_target_length = MAX_LENGTH

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_target_length, padding=padding, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=not None,
    )

    valid_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    valid_dataloader = DataLoader(
        valid_dataset, shuffle=True, collate_fn=data_collator, batch_size=1
    )
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=1)

    return valid_dataloader, test_dataloader


def prepare_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, tokenizer


def count_metrics(one_list, second_list):
    result = {}
    for metric in ['bleu', 'meteor', 'chrf', 'bertscore',
                   'calculate_wer_corpus', 'calculate_paraphrase_similarity']:
        result[metric] = ALL_METRICS[metric](one_list, second_list)
    result['wer'] = result.pop('calculate_wer_corpus')
    result['par.similarity'] = result.pop('calculate_paraphrase_similarity')
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--threshold", default=0.75, type=float)
    parser.add_argument("--max_iteration", default=100, type=int)
    parser.add_argument("--experiment_folder", default=Path('experiment/threshold_0.75/'), type=Path)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, tokenizer = prepare_model_and_tokenizer(device)
    _, test_dataloader = prepare_dataloader(tokenizer)

    attacker = GradientGuidedSearchStrategy(model, tokenizer, threshold=args.threshold, max_iteration = args.max_iteration)

    result = attacker.attack_dataset(test_dataloader, SENTENCES_TO_ATTACK, verbose=False)
    x_perturbed_with_brackets = [i[-1] for i in result]
    x_perturbed = [i.replace("[[", "").replace("]]", "") for i in x_perturbed_with_brackets]

    # translate perturbed text
    y_attacked = []
    for perturbed_id, perturbed_text in enumerate(tqdm(x_perturbed)):
        with torch.no_grad():
            translated = model.generate(torch.tensor(tokenizer.encode(perturbed_text)).unsqueeze(0).to(device))
            translated = tokenizer.decode(translated[0], skip_special_tokens=True)
        y_attacked.append(translated)

    # translate origin text
    x = []
    y = []
    y_without_attack = []
    iter = 0
    for batch in tqdm(test_dataloader):
        iter += 1
        if iter == SENTENCES_TO_ATTACK:
            break
        batch = {i: j.to(device) for i, j in batch.items()}
        with torch.no_grad():
            translated = model.generate(batch['input_ids'])
            translated = tokenizer.decode(translated[0], skip_special_tokens=True)
        x.append(tokenizer.decode(batch['input_ids'][0].tolist()))
        y.append(tokenizer.decode(batch['labels'][0].tolist()))
        y_without_attack.append(translated)
    x = [i.replace("▁", " ") for i in x]

    # save everything

    with open(f"{str(args.experiment_folder)}/x.json", 'w') as f:
        json.dump(x, f)

    with open(f"{str(args.experiment_folder)}/x_perturbed_with_brackets.json", 'w') as f:
        json.dump(x_perturbed_with_brackets, f)

    with open(f"{str(args.experiment_folder)}/x_perturbed.json", 'w') as f:
        json.dump(x_perturbed, f)

    with open(f"{str(args.experiment_folder)}/y.json", 'w') as f:
        json.dump(y, f)

    with open(f"{str(args.experiment_folder)}/y_without_attack.json", 'w') as f:
        json.dump(y_without_attack, f)

    with open(f"{str(args.experiment_folder)}/y_attacked.json", 'w') as f:
        json.dump(y_attacked, f)

    # count metrics
    
    orig_translate_metrics = count_metrics(y, y_without_attack)
    attack_translate_metrics = count_metrics(y, y_attacked)
    x_metrics = count_metrics(x, x_perturbed)

    table = {'orig.input': x, 'pert.input': x_perturbed_with_brackets,
             'labels': y, 'orig.translation': y_without_attack, 'pert.translation': y_attacked}

    for metric_name, metric_value in orig_translate_metrics.items():
        table[f"orig.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in attack_translate_metrics.items():
        table[f"attacked.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in x_metrics.items():
        table[f"x_{metric_name}"] = metric_value

    table = pd.DataFrame(table)

    table.to_csv(f"{str(args.experiment_folder)}/table.csv")
