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

# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
# from scipy.optimize import linear_sum_assignment

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--checkpoint_name', type=str, default='model.pkl')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
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
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--whole_env', action='store_true')
    parser.add_argument('--color', action='store_true')
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
    hparams['include_color'] = True
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

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]
    
    # if args.dataset == 'CIFAR10':
    #     train = 

    train_loaders = []

    # uda_loaders = [torch.utils.data.DataLoader(
    #     dataset=env,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [torch.utils.data.DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS, shuffle=False) for env in dataset]
    # eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    # eval_loader_names = ['env{}_in'.format(i)
    #     for i in range(len(in_splits))]
    # eval_loader_names += ['env{}_out'.format(i)
    #     for i in range(len(out_splits))]
    # eval_loader_names += ['env{}_uda'.format(i)
    #     for i in range(len(uda_splits))]

    # algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    # algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
    #     len(dataset) - len(args.test_envs), hparams)
    
    # if args.checkpoint:
    #     state_dict = torch.load(args.checkpoint+'/'+args.checkpoint_name)
    #     algorithm_dict = state_dict['model_dict']

    # if algorithm_dict is not None:
    #     algorithm.load_state_dict(algorithm_dict)

    # algorithm.to(device)

    #train_minibatches_iterator = zip(*train_loaders)
    train_minibatches_iterator = None
    #uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    model_dict = {'resnet':['18','34','50','101'],'resnet-d':['18','34'],'dinov2':['s14']}
    # algorithm.eval()
    tort_list = ['train','test']
    for tort,loader in enumerate(eval_loaders):
        for model in ['resnet-d','dinov2']:
            hparams['model'] = model
            if model == 'resnet':
                hparams['vit'] = 0
                hparams['dinov2'] = 0
                hparams['resnet'] = 1
                hparams['resnet50_augmix'] = 0
            elif model == 'resnet-d':
                hparams['resnet'] = 0
            elif model == 'dinov2':
                hparams['vit'] = 1
                hparams['dinov2'] = 1
            for size in model_dict[model]:
                hparams['arch'] = size
                os.makedirs(os.path.join(args.output_dir, f'{tort_list[tort]}/{model}_{size}/'),exist_ok=True)
                algorithm_class = algorithms.get_algorithm_class(args.algorithm)
                algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                    len(dataset) - len(args.test_envs), hparams)
                algorithm.to(device)
                algorithm.eval()
                features = []
                # preds = []
                true_labels = []
                # colors = []
                inds = []
                for i, (x,y,indices) in enumerate(loader):
                    x = x.to(device)
                    with torch.no_grad():
                        feat,predicted = algorithm.predict(x,get_pool=True)
                    features.append(feat.cpu())
                    # preds.append(predicted.cpu())
                    # features.append(feat)
                    # preds.append(predicted)
                    true_labels.append(y)
                    inds.append(indices)

                features = torch.cat(features,dim=0)
                print('Feature Shape: ',features.shape)
                true_labels = torch.cat(true_labels,dim=0)
                inds = torch.cat(inds,dim=0)

                torch.save(features,os.path.join(args.output_dir, f'{tort_list[tort]}/{model}_{size}/features.pt'))
                torch.save(true_labels,os.path.join(args.output_dir, f'{tort_list[tort]}/{model}_{size}/labels.pt'))
                torch.save(inds,os.path.join(args.output_dir, f'{tort_list[tort]}/{model}_{size}/indices.pt'))
            # features_np = torch.cat(features, dim=0).numpy()
            # preds_np = torch.cat(preds, dim=0).numpy()
            # true_labels_np = torch.cat(true_labels, dim=0).numpy()
            # #colors_np = torch.cat(colors, dim=0).numpy()
            # np.savez(os.path.join(args.output_dir, 'representations.npz'), representations=features_np, labels=true_labels_np, predictions=preds_np)

    print('done')