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
    parser.add_argument('--dc_dir', type=str)
    parser.add_argument('--dc_name', type=str,default='model.pkl')
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
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--tc_temperature', type=float, default=None)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--cal_dc', action='store_true')
    parser.add_argument('--nl_dc',action='store_true')
    parser.add_argument('--cal_teacher', action='store_true')
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
    env_i = args.test_envs[0]
    env = dataset[env_i]
    uda = []

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
    in_splits.append((in_, in_weights))
    out_splits.append((out, out_weights))
    if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]
    
    train_loaders = []

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [torch.utils.data.DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS, shuffle=False)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.teacher_algorithm)
    # ----- Load teachers ------
    teachers = []
    dir = args.teacher_dir
    for i in range(len(dataset)):
        if i in args.test_envs:
            teachers.append(None)
            continue
        path = dir+'/'+str(i)
        if args.cal_teacher:
            teacher_base = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
            teacher = algorithms.CalibratedAlgorithm(teacher_base)
        else:
            teacher = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
        state_dict = torch.load(path+'/model.pkl')
        teacher.load_state_dict(state_dict['model_dict'])
        if args.tc_temperature:
            if args.cal_teacher:
                teacher.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1)*teacher.temperature_scaling.temperature.item()*args.tc_temperature)
            else:
                teacher = algorithms.CalibratedAlgorithm(teacher)
                teacher.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1)*args.tc_temperature)
        teacher.eval()
        teacher.to(device)
        teachers.append(teacher)

    # ----- Load dc -----
    dc_hparams = hparams
    if args.nl_dc:
        dc_hparams['nonlinear_classifier'] = True
    if args.cal_dc:
        dc_base = algorithm_class(dataset.input_shape, len(dataset),
            len(dataset) - len(args.test_envs), dc_hparams)
        dc = algorithms.CalibratedAlgorithm(dc_base)
    else:
        dc = algorithm_class(dataset.input_shape, len(dataset),
            len(dataset) - len(args.test_envs), dc_hparams)
    dir = args.dc_dir
    state_dict = torch.load(dir+'/'+args.dc_name)
    dc.load_state_dict(state_dict['model_dict'])
    if args.temperature:
        if args.cal_dc:
            dc.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1)*dc.temperature_scaling.temperature.item()*args.temperature)
        else:
            dc = algorithms.CalibratedAlgorithm(dc)
            dc.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1)*args.temperature)
    dc.eval()
    dc.to(device)

    student_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = student_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), teachers, dc, args.test_envs[0],hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    #train_minibatches_iterator = zip(*train_loaders)
    train_minibatches_iterator = None
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def record_logits(model, output_file, dataloader, batch_size=64):
        all_data = []
        correct = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):
                count += len(x)
                x = x.to(device)
                y = y.to(device)
                tc_preds = []
                tc_max_probs = []
                tc_keys = []

                for i, teacher in enumerate(model.teachers):
                    if teacher is None:
                        continue
                    teacher.eval()
                    with torch.no_grad():
                        teacher_logits = teacher.predict(x)
                    teacher_output = torch.nn.functional.softmax(teacher_logits,dim=-1)
                    #teacher_output = F.softmax(teacher_logits,dim=-1)
                    max_prob = torch.max(teacher_output,dim=1).values
                    pred = torch.argmax(teacher_output,dim=1)
                    tc_preds.append(pred.tolist())
                    tc_max_probs.append(max_prob.tolist())
                    tc_keys.append(i)

                dc_pred = model.domain_classifier.predict(x)
                top_two_indices = torch.topk(dc_pred, 2, dim=1).indices
                top_two_indices_list = top_two_indices.tolist()
                
                logits = model.teachers_outputs(x)
                pred = torch.argmax(logits,dim=1)
                correct += (pred == y).sum().item()

                logits_list = logits.tolist()
                y_list = y.tolist()
                for i in range(len(x)):
                    all_data.append([idx * batch_size + i, y_list[i]] + logits_list[i] + top_two_indices_list[i] + [tc_pred[i] for tc_pred in tc_preds] + [tc_max_prob[i] for tc_max_prob in tc_max_probs])

        print('Acc: ', str(correct/count))
        # Write to CSV file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'ground_truth'] + [f'logit_{i}' for i in range(len(logits_list[0]))] + ['dc_top1', 'dc_top2'] + [f'tc_{i}_pred' for i in tc_keys] + [f'tc_{i}_max_prob' for i in tc_keys])
            writer.writerows(all_data)

    output_file = os.path.join(args.output_dir,'prob.csv')
    record_logits(algorithm, output_file, eval_loaders[0])
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
