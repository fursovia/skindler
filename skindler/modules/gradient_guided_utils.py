from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from skindler import MODEL_NAME, DATASET_NAME, MAX_LENGTH
from skindler.modules.metrics import ALL_METRICS


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
