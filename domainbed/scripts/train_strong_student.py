# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
from sklearn.metrics import accuracy_score

from domainbed import datasets_nlp as datasets
from domainbed import hparams_registry

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Convert logits to predicted class labels (max probability)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    return {"accuracy": accuracy}

class WeakLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.data = pd.read_csv(data_file, delimiter="\t")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['weak_label']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train strong student')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="SST2")
    parser.add_argument('--model_name', type=str, default="bert-base-uncased")
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--epoch', type=int, default=3,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="student_train_output")
    parser.add_argument('--SGD', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + '/logs', exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams("ERM", args.dataset)
    else:
        hparams = hparams_registry.random_hparams("ERM", args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    weak_label_file = os.path.join(args.data_dir, "SST-2/SST-2", "train_weak_labeled.tsv")
    train_dataset = WeakLabeledDataset(weak_label_file, tokenizer, args.max_length)

    if args.SGD:
        # Define the optimizer (SGD with momentum)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams['lr'],  # Learning rate
            momentum=hparams['momentum'] if 'momentum' in hparams else 0.0,  # Momentum
            weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
        )
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=hparams['batch_size'],
            per_device_eval_batch_size=64,
            num_train_epochs=args.epoch,
            learning_rate=hparams['lr'],
            weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
            logging_dir=args.output_dir + '/logs',
            logging_steps=1000,
            save_total_limit=2,
            report_to="tensorboard",
            disable_tqdm=True,
            lr_scheduler_type='linear',
            warmup_steps=hparams['warmup_steps'] if 'warmup_steps' in hparams else 500,
            gradient_accumulation_steps=1,
        )

        # Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )

    else:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=hparams['batch_size'],
            per_device_eval_batch_size=64,
            num_train_epochs=args.epoch,
            learning_rate=hparams['lr'],
            weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
            logging_dir=args.output_dir + '/logs',
            logging_steps=1000,
            save_total_limit=2,
            report_to="tensorboard",
            disable_tqdm=True,
            warmup_steps=hparams['warmup_steps'] if 'warmup_steps' in hparams else 500,
        )

        # Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics
        )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results for {args.model_name}:", eval_results)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')