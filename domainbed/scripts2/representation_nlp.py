# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import pandas as pd
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
import numpy as np
#from sklearn.metrics import accuracy_score

from domainbed.datasets_nlp import SST2TorchDataset
from domainbed import hparams_registry
#from domainbed import algorithms
from domainbed.lib import misc
#from domainbed.loss import logconf_loss_fn
#from domainbed import networks
#from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, InfiniteDataLoaderWithoutReplacement
#from domainbed.intrinsic import intrinsic_dimension, intrinsic_dimension_said

access_token = ''

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
    # if args.dataset in vars(datasets):
    #     dataset = vars(datasets)[args.dataset](path=args.data_dir,file_name=args.data_name,tokenizer_name=args.model_name,split_ratio=[args.n, args.N, args.test_length],max_length=args.max_length,seed=args.seed,flip_prob=args.flip_prob)
    # else:
    #     raise NotImplementedError

    df = pd.read_csv(os.path.join(args.data_dir, args.data_name), delimiter="\t")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,token=access_token)
    dataset = SST2TorchDataset(df,tokenizer,80)

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "3600"
    #algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    
    if args.checkpoint:
        model = AutoModel.from_pretrained(args.checkpoint,output_hidden_states=True)
    else:
        model = AutoModel.from_pretrained(args.model_name, output_hidden_states=False,token=access_token)

    if 'gpt2' in args.model_name:
        model.config.pad_token_id = model.config.eos_token_id

    if args.freeze:
        # Freeze all layers except for the classification head
        for name, param in model.named_parameters():
            if "classifier" not in name:  # "classifier" is the head of the model
                param.requires_grad = False

    # # Intrinsic Dimension
    # if args.intrinsic_dimension:
    #     if args.said:
    #         model = intrinsic_dimension_said(model,args.intrinsic_dimension,None,set(),"fastfood", device=device)
    #     else:
    #         model = intrinsic_dimension(model,args.intrinsic_dimension,None,set(),"fastfood", device=device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("***** Model Trainable Parameters {} *****".format(trainable_params))
    
    dataloader = DataLoader(dataset, batch_size=hparams['batch_size'], num_workers=0, shuffle=False)


    # 训练模型
    num_epochs = args.epoch
    model.to(device)
    model.eval() #turn off dropout for reproducibility
    features = []
    true_labels = []
    inds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            indx = batch['idx']
            #labels = torch.nn.functional.one_hot(labels,num_classes=2).float()
            #print(input_ids.shape)
            #print(attention_mask.shape)
            #optimizer.zero_grad()
            feats = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            if 'bert' not in args.model_name:
                feats = feats[:,-1,:]
            else:
                feats = feats[:,0,:]
            features.append(feats.cpu())
            true_labels.append(labels)
            inds.append(indx)

    features = torch.cat(features,dim=0)
    print('Feature Shape: ',features.shape)
    true_labels = torch.cat(true_labels,dim=0)
    inds = torch.cat(inds,dim=0)

    os.makedirs(args.output_dir+f'/{args.model_name}',exist_ok=True)
    torch.save(features,os.path.join(args.output_dir, f'{args.model_name}/features.pt'))
    torch.save(true_labels,os.path.join(args.output_dir, f'{args.model_name}/labels.pt'))
    torch.save(inds,os.path.join(args.output_dir, f'{args.model_name}/indices.pt'))            

    print('Training finished.')

    #print(f"Evaluation results for {args.model_name}:", eval_results)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')