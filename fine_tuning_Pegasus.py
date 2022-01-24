# -*- coding: utf-8 -*-
"""Fine_tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sJRjQJO8Su8kwheVSP1QgLGRDg0tATM0
SOURCE:
    https://gist.github.com/jiahao87/50cec29725824da7ff6dd9314b53c4b3
"""

from datasets import load_from_disk
import torch
from transformers import PegasusModel, PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments

class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)


def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts=None, val_labels=None,
                 test_texts=None, test_labels=None):
    """
    Prepare input data for model fine-tuning
    """
    tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir="HF_HOME")
    prepare_val = False if val_texts is None or val_labels is None else True
    prepare_test = False if test_texts is None or test_labels is None else True

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True)
        decodings = tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
    test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

    return train_dataset, val_dataset, test_dataset, tokenizer


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False,
                        output_dir='./pegasus_large_fine_tune/results'):
    """
    Prepare configurations and base model for fine-tuning
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir="HF_HOME").to(torch_device)

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if val_dataset is not None:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=1,  # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=8,  # batch size for evaluation, can increase if memory allows
            save_steps=5000,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            lr_scheduler_type="cosine",
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            eval_steps=100,  # number of update steps before evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0,  # strength of weight decay
            learning_rate=6.35e-05,
            logging_dir='./pegasus_large_fine_tune/logs',  # directory for storing logs
            logging_steps=50,
            push_to_hub=True,
            hub_model_id="SophieTr/fine-tune-Pegasus-large",
            report_to="wandb",
            load_best_model_at_end=True,
            fp16=True,
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            tokenizer=tokenizer
        )

    else:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=1,  # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training, can increase if memory allows
            save_steps=5000,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            lr_scheduler_type="cosine",
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0,  # strength of weight decay
            logging_dir='./pegasus_large_fine_tune/logs',  # directory for storing logs
            logging_steps=10,
            learning_rate= 6.35e-05,
            push_to_hub=True,
            hub_model_id="SophieTr/fine-tune-Pegasus-large",
            report_to="wandb",
            load_best_model_at_end=True,
            fp16=True,
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            tokenizer=tokenizer
        )

    return trainer


if __name__ == '__main__':
    dataset = load_from_disk("reddit_clean")
    train_texts, train_labels = dataset['train']['content'], dataset['train']['summary']
    val_texts, val_labels = dataset['valid']['content'], dataset['valid']['summary']
    test_texts, test_labels = dataset['test']['content'], dataset['test']['summary']

    model_name = 'google/pegasus-large'  # 'google/pegasus-large'
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts,
                                                                       val_labels, test_texts, test_labels)
    print("First in train dataset: ", train_dataset[0].shape)
    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset)

    trainer.train()
    trainer.evaluate(test_dataset)
    trainer.push_to_hub()

## TO DO: push model to HF Hub
