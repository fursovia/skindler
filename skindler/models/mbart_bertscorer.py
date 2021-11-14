import torch
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from transformers import MBart50TokenizerFast, Trainer, default_data_collator
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.mbart.modeling_mbart import MBartEncoder
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from skindler import MBART_NAME, MAX_LENGTH


class MbartBertScorer(torch.nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder: MBartEncoder = MBartForConditionalGeneration.from_pretrained(model_name).get_encoder().eval()
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.dropout = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(1024 * 2, 256)
        self.linear2 = torch.nn.Linear(256, 1)
        self.loss = torch.nn.L1Loss()

    def get_embeddings(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = outputs.last_hidden_state
        return embeddings

    def get_logits(self, embeddings):
        embeddings = torch.cat(
            (
                torch.sum(embeddings, dim=1),
                torch.max(embeddings, dim=1).values,
            ),
            dim=1
        )
        embeddings = self.dropout(embeddings)
        embeddings = self.linear1(embeddings)
        embeddings = torch.relu(embeddings)
        logits = self.linear2(embeddings)
        return logits

    def forward(
            self,
            input_ids=None,
            bertscore=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        embeddings = self.get_embeddings(input_ids, attention_mask)
        logits = self.get_logits(embeddings)

        loss = None
        if bertscore is not None:
            loss = self.loss(logits.view(-1), bertscore.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


if __name__ == '__main__':
    args = {
        'output_dir': '../skindler_data/experiments/mbart_bertscorer_2',
        'cache_dir': '../skindler_data/cache',
        'model_name': MBART_NAME,
        'text_column_name': 'x',

    }
    data_files = {
        "train": "../skindler_data/mbart_bertscore/mbart_bertscore_train.json",
        "validation": "../skindler_data/mbart_bertscore/mbart_bertscore_validation.json"
    }
    training_args = TrainingArguments(
        output_dir=args['output_dir'],
        label_names=['input_ids'],
        report_to=['wandb'],
        run_name='mbart_bertscorer_l1_loss',
        save_total_limit=2,
        dataloader_num_workers=4,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        do_train=True,
        do_eval=True,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=5_000,
        save_steps=5_000,
        learning_rate=0.003,
        num_train_epochs=5
    )
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=args['cache_dir'])
    column_names = raw_datasets["train"].column_names
    column_names.remove('bertscore')
    tokenizer = MBart50TokenizerFast.from_pretrained(args['model_name'])

    print(raw_datasets)


    def tokenize_function(examples):
        return tokenizer(
            examples[args['text_column_name']],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
        )


    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            desc="Running tokenizer on every text in dataset",
        )

    print('load model')
    model = MbartBertScorer(args['model_name'])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    print('start training')
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
