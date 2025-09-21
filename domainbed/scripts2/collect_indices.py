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
import torchvision.models as models
from torch.utils.data import DataLoader

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import networks

#import simsiam.builder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    # parser.add_argument('--teacher_dir', type=str)
    parser.add_argument('--dc_dir', type=str)
    # parser.add_argument('--dc_name',type=str,default='model.pkl')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="EfficientStudent")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_algorithm', type=str, default='ERM')
    parser.add_argument('--dc_algorithm', type=str, default='ERM')
    parser.add_argument('--tc_algorithm', type=str, default='ERM')
    parser.add_argument('--checkpoint_name', type=str, default='model.pkl')
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="indices")
    parser.add_argument('--output_name', type=str, default="indices.json")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--tc_temperature', type=float, default=None)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--generalization', action='store_true')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--dinov', action='store_true')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--all_envs', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = args.start_step
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
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

    hparams['cross_entropy'] = False
    hparams['index_dataset'] = True
    if args.temperature:
        hparams['tc_temperature'] = args.temperature
        hparams['dc_temperature'] = args.temperature
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    print('Dataset Loaded')
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    domain_len = []
    for env_i, env in enumerate(dataset):
        print(f'Splitting env_{env_i}')
        uda = []
        domain_len.append(len(env))

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        
        if env_i in args.test_envs:
            uda_splits.append((in_,in_weights))
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    print('Getting indices')
    result = [[] for i in range(dataset.num_classes)]
    if args.train_only:
        if args.all_envs:
            for j, (env,_) in enumerate(in_splits):
                print(f'Start env_{j}')
                result = [[] for i in range(dataset.num_classes)]
                # Create DataLoader with a large batch size
                # dataloader = FastDataLoader(dataset=env, batch_size=512, num_workers=8)

                # for batch in dataloader:
                #     indices, labels = batch[2], batch[1]
                #     for idx, label in zip(indices, labels):
                #         result[label.item()].append(idx.item())
                for i, (image,label,index) in enumerate(env):
                    result[label].append(i)
                dir = args.output_dir +'/'+str(j)
                os.makedirs(dir, exist_ok=True)
                path = dir+'/'+args.output_name
                with open(path,'w') as file:
                    json.dump(result,file)
                
        else:
            for (env,_) in uda_splits:
                for i, (image,label,index) in enumerate(env):
                    result[label].append(i)
    else:
        for (env,_) in uda_splits:
            for i, (image,label,index) in enumerate(env):
                result[label].append(index)
    if not args.all_envs:
        dir = args.output_dir
        path = dir+'/'+args.output_name
        with open(path,'w') as file:
            json.dump(result,file)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
