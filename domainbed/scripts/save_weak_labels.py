# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save weak labels')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="SST2")
    parser.add_argument('--model_name', type=str, default="bert-base-uncased")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](path=args.data_dir, tokenizer_name=args.model_name, max_length=args.max_length, seed=args.seed)
    else:
        raise NotImplementedError

    if args.checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    else:
        raise ValueError("Checkpoint path is required to load the weak teacher model.")

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load original training data
    train_data = pd.read_csv(os.path.join(args.data_dir, "SST-2/SST-2", "train.tsv"), delimiter="\t")

    weak_labels = []
    for text in train_data['sentence']:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = logits.argmax(dim=-1).item()
            weak_labels.append(prediction)

    train_data['weak_label'] = weak_labels

    # Save weak labeled data
    weak_label_file = os.path.join(args.data_dir, "SST-2/SST-2", "train_weak_labeled.tsv")
    train_data.to_csv(weak_label_file, sep='\t', index=False)

    print(f"Weak labels saved to {weak_label_file}")