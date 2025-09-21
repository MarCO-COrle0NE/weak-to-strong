# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import csv

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--teacher_dir', type=str)
    parser.add_argument('--teacher_name', type=str,default='model.pkl')
    parser.add_argument('--dc_dir', type=str)
    parser.add_argument('--dc_name',type=str,default='model.pkl')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--teacher_algorithm', type=str, default="ERM")
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
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
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
    hparams['index_dataset'] = True
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

    env_i = args.test_envs[0]
    env = dataset[env_i]

    eval_loaders = [torch.utils.data.DataLoader(
        dataset=dataset[i],
        batch_size=1,
        num_workers=dataset.N_WORKERS, shuffle=False)
        for i in range(len(dataset))]

    algorithm_class = algorithms.get_algorithm_class(args.teacher_algorithm)
    # ----- Load teachers ------
    teachers = []
    dir = args.teacher_dir
    for i in range(len(dataset)):
        # if i in args.test_envs:
        #     teachers.append(None)
        #     continue
        path = dir+'/'+str(i)
        teacher = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
        state_dict = torch.load(path+'/'+args.teacher_name)
        teacher.load_state_dict(state_dict['model_dict'])
        teacher.eval()
        teacher.to(device)
        teachers.append(teacher)

    # ----- Load dc -----
    dc_hparams = hparams
    dc_hparams['vit'] = False
    dc_hparams['resnet18'] = False
    dc_hparams['resnet50_augmix'] = True
    dc = algorithm_class(dataset.input_shape, len(dataset),
        len(dataset) - len(args.test_envs), dc_hparams)
    dir = args.dc_dir
    state_dict = torch.load(dir+'/'+args.dc_name)
    dc.load_state_dict(state_dict['model_dict'])
    dc.eval()
    dc.to(device)

    def record_logits(dc, teachers, output_file, dataloaders):
        all_data = [[None for i in range(len(dataloaders[j]))] for j in range(len(dataloaders))]
        with torch.no_grad():
            for j, dataloader in enumerate(dataloaders):
                for x, y, i in dataloader:
                    data = {'dc':None,'tc':[]}
                    x = x.to(device)
                    dc_pred = dc.predict(x).tolist()
                    if len(dc_pred) == 1 and isinstance(dc_pred[0],list):
                        dc_pred = dc_pred[0]
                    data['dc'] = dc_pred
                    for teacher in teachers:
                        tc_pred = teacher.predict(x).tolist()
                        if len(tc_pred) == 1 and isinstance(tc_pred[0],list):
                            tc_pred = tc_pred[0]
                        data['tc'].append(tc_pred)
                    all_data[j][i] = data

        # Write to json file
        with open(output_file,'w') as file:
            json.dump(all_data,file)
        
    output_file = os.path.join(args.output_dir,'diction.json')
    record_logits(dc,teachers, output_file, eval_loaders)
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
