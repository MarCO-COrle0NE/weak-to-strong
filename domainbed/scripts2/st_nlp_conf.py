# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification
import numpy as np
#from sklearn.metrics import accuracy_score

from domainbed import datasets_nlp as datasets
from domainbed import hparams_registry
#from domainbed import algorithms
from domainbed.lib import misc
from domainbed.loss import logconf_loss_fn
#from domainbed import networks
#from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, InfiniteDataLoaderWithoutReplacement
from domainbed.intrinsic import intrinsic_dimension, intrinsic_dimension_said

# Define compute_metrics function
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)  # Convert logits to predicted class labels (max probability)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(labels, preds)
    
#     return {"accuracy": accuracy}

def accuracy(model,dataloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask).logits
                correct += (logits.argmax(1).eq(labels).float()).sum().item()
                total += len(labels)
    return correct / total



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
    #parser.add_argument('--flip_st',action='store_true')
    parser.add_argument('--freeze',action='store_true')
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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'/logs', exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

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

    print(dataset.prior)
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "3600"
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
    
    train_dataloader = DataLoader(dataset[1], batch_size=hparams['batch_size'], num_workers=2, shuffle=True)
    test_dataloader = DataLoader(dataset[2], batch_size=hparams['batch_size'], num_workers=2, shuffle=False)
    loss_function = logconf_loss_fn(prior=dataset.prior)

    # 初始化余弦退火学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    nsteps = len(dataset[0]) * args.epoch // hparams['batch_size']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)

    # 训练模型
    num_epochs = args.epoch
    model.to(device)
    model.eval() #turn off dropout for reproducibility
    step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = torch.nn.functional.one_hot(labels,num_classes=2).float()

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask).logits
            loss = loss_function(logits, labels, step_frac=step/nsteps)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % 100 == 0 and step % (nsteps // num_epochs) > 0:
                print(f'Epoch {step / (nsteps // num_epochs)}, Loss: {running_loss / (step % (nsteps // num_epochs) * hparams['batch_size'])}')

        results = {
                'step': step,
                'epoch': epoch,
                'loss':running_loss / len(train_dataloader),
            }
        train_acc = accuracy(model,train_dataloader,device)
        test_acc = accuracy(model,test_dataloader,device)
        results['eval_train_accuracy'] = train_acc
        results['eval_test_accuracy'] = test_acc
        results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
        #print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}')
        print(results)

    print('Training finished.')

    #print(f"Evaluation results for {args.model_name}:", eval_results)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')