import torch
from transformers.models.marian.modeling_marian import MarianEncoder
from transformers import MarianTokenizer, Trainer, default_data_collator
from transformers.training_args import TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.trainer_utils import IntervalStrategy
from transformers import EarlyStoppingCallback
from datasets import load_dataset

from skindler import MODEL_NAME, MAX_LENGTH


class MarianAutoEncoder(torch.nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = MarianEncoder.from_pretrained(model_name).eval()
        self.num_labels = self.encoder.config.vocab_size
        self.linear = torch.nn.Linear(512, self.num_labels)
        self.dropout = torch.nn.Dropout(dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        calculate_loss=True,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            embeddings = outputs.last_hidden_state

        embeddings = self.dropout(embeddings)
        logits = self.linear(embeddings)

        loss = None
        if calculate_loss:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, input_ids.view(-1), torch.tensor(self.loss.ignore_index).type_as(input_ids)
                )
                loss = self.loss(active_logits, active_labels)
            else:
                loss = self.loss(logits.view(-1, self.num_labels), input_ids.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


if __name__ == '__main__':
    args = {
        'output_dir': './ae_logs',
        'cache_dir': 'cache',
        'model_name': MODEL_NAME,
        'text_column_name': 'en',

    }
    data_files = {"train": "data/train.json", "validation": "data/valid.json"}
    training_args = TrainingArguments(
        output_dir=args['output_dir'],
        report_to=['wandb'],
        save_total_limit=10,
        dataloader_num_workers=4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        do_train=True,
        do_eval=True,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=500,
        save_steps=500,
        learning_rate=0.003,
    )
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=args['cache_dir'])
    column_names = raw_datasets["train"].column_names
    tokenizer = MarianTokenizer.from_pretrained(args['model_name'])

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

    model = MarianAutoEncoder(args['model_name'])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
