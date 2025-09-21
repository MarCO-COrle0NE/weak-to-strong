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

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import networks

import simsiam.builder

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
    parser.add_argument('--output_dir', type=str, default="train_output")
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

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]
    train_loaders = None

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    # eval_loader_names += ['env{}_uda'.format(i)
    #     for i in range(len(uda_splits))]

    #algorithm_class = algorithms.get_algorithm_class(args.tc_algorithm)
    #algorithm_class2 = algorithms.get_algorithm_class('CalibratedAlgorithm')
    
    with open(args.dc_dir,'r') as file:
        all_data = json.load(file)

    student_classes = ['WeightedOutput',
        'WeightedLoss',
        'Greedy',
        'MajorityVoting',
        'Average',
        'WeightedVoting']
    student_class = algorithms.get_algorithm_class(args.algorithm)
    if args.checkpoint:
        base_dict = torch.load(args.checkpoint+'/'+args.checkpoint_name)
        if args.checkpoint_algorithm in student_classes:
            algorithm = student_class(dataset.input_shape, dataset.num_classes,
                len(dataset) - len(args.test_envs), all_data, args.test_envs[0], hparams, domain_len=domain_len)
            algorithm.load_state_dict(base_dict['model_dict'])
        else:
            if not args.dinov:
                algorithm_base = simsiam.builder.SimSiam(
                models.__dict__[base_dict['arch']],
                2048, 512)
                algorithm_base.load_state_dict({k.replace('module.', ''): v for k, v in base_dict['state_dict'].items()})
                algorithm = student_class(dataset.input_shape, dataset.num_classes,
                len(dataset) - len(args.test_envs), all_data, args.test_envs[0], hparams, featurizer = algorithm_base.encoder, domain_len=domain_len)
            else:
                algorithm_base = networks.DinoV2(dataset.input_shape,hparams)
                algorithm_base.network.load_state_dict({k.replace('module.backbone.', ''): v for k, v in base_dict['student'].items() if 'backbone' in k})
                algorithm = student_class(dataset.input_shape, dataset.num_classes,
                len(dataset) - len(args.test_envs), all_data, args.test_envs[0], hparams, featurizer = algorithm_base, domain_len=domain_len)
    else:
        algorithm = student_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), all_data, args.test_envs[0], hparams, domain_len=domain_len)
    if args.freeze:
        algorithm.freeze()
    algorithm.to(device)

    train_minibatches_iterator = None
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    generalization = args.generalization
    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        if generalization:
            minibatches_device = [(x.to(device), y.to(device), i.to(device))
                for x,y,i in next(train_minibatches_iterator)]
        else:
            minibatches_device = None
        if args.task == "domain_adaptation":
            uda_device = [(x.to(device),i.to(device))
                for x,_,i in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device, indexed=True)
                results[name+'_acc'] = acc

            agreement, agreement_neg, agreement_pos = misc.agreement(algorithm, eval_loaders[-1], weights, device, indexed=True)
            results['agreement'] = agreement
            results['agreement_neg'] = agreement_neg
            results['agreement_pos'] = agreement_pos
            
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
