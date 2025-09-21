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

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import torch
import torch.nn as nn
import torch.optim as optim

# Temperature scaling module
class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

# Function to calibrate the model using temperature scaling
def calibrate_model(model, val_loaders, lr = 0.01, max_iter=1000,checkpoint_freq=100):
    model.eval()
    temperature_scaling = TemperatureScaling().cuda()

    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for val_loader in val_loaders:
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                logits = model.predict(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    losses = []

    optimizer = optim.LBFGS([temperature_scaling.temperature], lr=lr, max_iter=max_iter)

    def eval():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(temperature_scaling(logits), labels)
        loss.backward()
        losses.append(loss)
        return loss

    optimizer.step(eval)
    losses=losses[::checkpoint_freq]

    return temperature_scaling, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_algorithm', type=str, default='ERM')
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
    parser.add_argument('--checkpoint_freq', type=int, default=5,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--cal_teachers', action='store_true')
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
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = None

    cal_loaders = [torch.utils.data.DataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)]

    uda_loaders = None

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.cal_teachers:
        paths = []
        loaders = []
        output_dirs = []
        for i in range(len(dataset)):
            paths.append(args.checkpoint+'/'+str(i))
            loaders.append([cal_loaders[i]])
            output_dirs.append(args.output_dir+'/'+str(i))
    else:
        paths = [args.checkpoint]
        loaders = [cal_loaders]
        output_dirs = [args.output_dir]
    for i in range(len(paths)):
        path = paths[i]
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)
        state_dict = torch.load(path+'/'+args.checkpoint_name)
        algorithm_dict = state_dict['model_dict']
        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)

        algorithm.to(device)
        # Calibrate the model
        temperature_scaling, losses = calibrate_model(algorithm,loaders[i],args.lr,args.max_iter,args.checkpoint_freq)

        calibrated_algorithm = algorithms.CalibratedAlgorithm(algorithm,temperature_scaling)

        def save_checkpoint(output_dir,filename):
            if args.skip_model_save:
                return
            save_dict = {
                "args": vars(args),
                "model_input_shape": dataset.input_shape,
                "model_num_classes": dataset.num_classes,
                "model_num_domains": len(dataset) - len(args.test_envs),
                "model_hparams": hparams,
                "model_dict": calibrated_algorithm.state_dict()
            }
            torch.save(save_dict, os.path.join(output_dir, filename))

        print('Temperature:',temperature_scaling.temperature)
        print('Losses:',losses)
        os.makedirs(output_dirs[i], exist_ok=True)
        save_checkpoint(output_dirs[i],'model.pkl')
        results = {}
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        for name, loader, weights in evals:
            acc = misc.accuracy(algorithm, loader, weights, device)
            results[name+'_acc'] = acc

        results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

        results_keys = sorted(results.keys())
        misc.print_row(results_keys, colwidth=12)
        misc.print_row([results[key] for key in results_keys],
                    colwidth=12)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
