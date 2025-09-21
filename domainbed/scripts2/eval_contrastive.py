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
from sklearn.random_projection import GaussianRandomProjection
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.losses.self_supervised_loss import SelfSupervisedLoss
from pytorch_metric_learning.distances import LpDistance

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
    parser.add_argument('--temperature', type=float, default=0.5) 
    parser.add_argument('--n_components', type=int, default=None) 
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
    parser.add_argument('--euclidean', action='store_true')
    parser.add_argument('--project',action='store_true')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--deterministic_only',action='store_true')
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

    hparams['index_dataset'] = False
    hparams['contrastive'] = True
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

    eval_loaders = [torch.utils.data.DataLoader(
        dataset=dataset[args.test_envs[0]],
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS, shuffle=False)]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint+'/'+args.checkpoint_name)
        algorithm_dict = state_dict['model_dict']

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    #train_minibatches_iterator = zip(*train_loaders)
    train_minibatches_iterator = None
    checkpoint_vals = collections.defaultdict(lambda: [])
    if not args.euclidean:
        contrastive_loss = misc.SpectralContrastiveLoss(mu=1.0,eval=True)
    else:
        distance = LpDistance(normalize_embeddings=False, p=2)
        base_loss = NTXentLoss(temperature=args.temperature,distance=distance)
        contrastive_loss = SelfSupervisedLoss(base_loss)

    if args.project:
        if args.n_components:
            projector = GaussianRandomProjection(n_components=args.n_components,random_state=args.seed)
        else:
            projector = GaussianRandomProjection(n_components=dataset.num_classes,random_state=args.seed)

    algorithm.eval()
    with torch.no_grad():
        features1 = []
        features2 = []
        for i, (x,y) in enumerate(eval_loaders[0]):
            x1,x2=x
            x1 = x1.to(device)
            x2 = x2.to(device)
            feat1,predicted = algorithm.predict(x1,get_pool=True)
            feat2,predicted = algorithm.predict(x2,get_pool=True)
            features1.append(feat1)
            features2.append(feat2)

        features1 = torch.cat(features1,dim=0)
        features2 = torch.cat(features2,dim=0)
        if args.project:
            features1 = projector.fit_transform(features1.cpu())
            features2 = projector.fit_transform(features2.cpu())
            features1 = torch.from_numpy(features1)
            features2 = torch.from_numpy(features2)
        if args.euclidean:
            loss = contrastive_loss(features1,features2)
        else:
            loss,info = contrastive_loss(features1,features2)
    
    loss = loss.mean().item()
    result = {'size':hparams['arch'],'loss': loss, 'euclidean':1, 'seed':args.seed, 'checkpoint':args.checkpoint}
    print('Contrastive Loss: ', loss)
    if not args.euclidean:
        result['euclidean']=0
        loss1 = info['part1'].mean().item()
        loss2 = info['part2'].mean().item()
        result['part1'] = loss1
        result['part2'] = loss2
        print('Contrastive Loss part1: ', loss1)
        print('Contrastive Loss part2: ', loss2)

    with open(os.path.join(args.output_dir, 'contrastive_loss.jsonl'), 'a') as f:
        #json.dump(result,f)
        f.write(json.dumps(result, sort_keys=True) + "\n")

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
