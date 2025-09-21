# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import datetime

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
from accelerate import Accelerator

from domainbed import datasets_nlp as datasets
from domainbed import hparams_registry
#from domainbed import algorithms
from domainbed.lib import misc
#from domainbed import networks
#from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, InfiniteDataLoaderWithoutReplacement
from domainbed.intrinsic import intrinsic_dimension, intrinsic_dimension_said

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Convert logits to predicted class labels (max probability)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    return {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_name', type=str, default='train.tsv')
    parser.add_argument('--dataset', type=str, default="SST2")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--model_name', type=str, default="bert-base-uncased")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_name', type=str, default='model.pkl')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--n', type=int, default=3000,
        help='Seed for everything else')
    parser.add_argument('--N', type=int, default=30000,
        help='Seed for everything else')
    parser.add_argument('--test_length', type=int, default=6000,
        help='Seed for everything else')
    parser.add_argument('--max_length', type=int, default=300,
        help='Seed for everything else')
    parser.add_argument('--epoch', type=int, default=3,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[2])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--intrinsic_dimension', type=int, default=None)
    parser.add_argument('--flip_prob', type=float, default=None)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--save_labels',action='store_true')
    parser.add_argument('--said',action='store_true')
    parser.add_argument('--SGD',action='store_true')
    parser.add_argument('--strict',action='store_true')
    parser.add_argument('--deterministic_only',action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    #start_step = args.start_step
    algorithm_dict = None
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    timeout = datetime.timedelta(seconds=3600)
    dist.init_process_group(backend="nccl", timeout=timeout)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'/logs', exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.deterministic_only:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](path=args.data_dir,file_name=args.data_name,tokenizer_name=args.model_name,split_ratio=[args.n, args.N, args.test_length],max_length=args.max_length,seed=args.seed,flip_prob=args.flip_prob)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "3600"
    accelerator = Accelerator()
    #algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    
    if args.checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=dataset.num_classes)

    if 'gpt2' in args.model_name:
        model.config.pad_token_id = model.config.eos_token_id

    if args.freeze:
        # Freeze all layers except for the classification head
        for name, param in model.named_parameters():
            if "classifier" not in name:  # "classifier" is the head of the model
                param.requires_grad = False

    # Intrinsic Dimension
    if args.intrinsic_dimension:
        if args.said:
            model = intrinsic_dimension_said(model,args.intrinsic_dimension,None,set(),"fastfood", device=device)
        else:
            model = intrinsic_dimension(model,args.intrinsic_dimension,None,set(),"fastfood", device=device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("***** Model Trainable Parameters {} *****".format(trainable_params))

    if args.SGD:
        # Define the optimizer (SGD with momentum)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams['lr'],            # Learning rate
            momentum=hparams['momentum'] if 'momentum' in hparams else 0.0,       # Momentum
            weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
        )
    # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=hparams['batch_size'],
            per_device_eval_batch_size=64,
            num_train_epochs=args.epoch-args.start_epoch,
            learning_rate=hparams['lr'], 
            weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
            logging_dir=args.output_dir+'/logs',
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
            train_dataset=dataset[1],
            eval_dataset={'teacher':dataset[0],'student':dataset[1],'test':dataset[2]},
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )

    else:
        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=hparams['lr'],            # Learning rate
        #     # weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
        # )
        # nsteps = len(dataset[1]) * args.epoch // hparams['batch_size']
        # scheduler = CosineAnnealingLR(optimizer,nsteps)
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=hparams['batch_size'],
            per_device_eval_batch_size=hparams['batch_size'],
            gradient_accumulation_steps=hparams['gradient_accumulation_steps'] if 'gradient_accumulation_steps' in hparams else 1,
            num_train_epochs=args.epoch-args.start_epoch,
            learning_rate=hparams['lr'], 
            weight_decay=hparams['weight_decay'] if 'weight_decay' in hparams else 0,
            logging_dir=args.output_dir+'/logs',
            lr_scheduler_type='cosine',
            logging_steps=100,
            save_total_limit=2,
            report_to="tensorboard",
            disable_tqdm=True,
            warmup_steps=hparams['warmup_steps'] if 'warmup_steps' in hparams else 0,
            dataloader_num_workers=1
        )

        # Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset[1],
            eval_dataset={'teacher':dataset[0],'student':dataset[1],'test':dataset[2]},
            compute_metrics=compute_metrics,
            #optimizers=(optimizer,scheduler)
        )

    model, trainer = accelerator.prepare(model, trainer)

    try:
        # Training code
        trainer.train()
         # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Evaluation results for {args.model_name}:", eval_results)
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        # Destroy the process group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    if args.save_labels:
            import pandas as pd
            # Save weak labels
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)

            # Load original training data
            train_data = pd.read_csv(os.path.join(args.data_dir, "train.tsv"), delimiter="\t")

            train_dataset = dataset[0]  # Assuming dataset[0] is the training dataset
            dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'],shuffle=False)

            weak_labels = []
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predictions = logits.argmax(dim=-1).cpu().tolist()
                    weak_labels.extend(predictions)

            train_data['weak_label'] = weak_labels

            # Save weak labeled data
            weak_label_file = os.path.join(args.data_dir,f"{args.model_name}_{args.n}.tsv")
            train_data.to_csv(weak_label_file, sep='\t', index=False)

            print(f"Weak labels saved to {weak_label_file}")
