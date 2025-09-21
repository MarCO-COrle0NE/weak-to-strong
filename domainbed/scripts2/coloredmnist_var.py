import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import argparse
#import wandb
import logging
import pickle

root_dir = r'../..'
SEED = 42
TOL_FP = 1e-12

plt.rc('font', size=18)#weight='bold', 
plt.rc('legend', fontsize=18)
plt.rc('lines', linewidth=3, markersize=9)
mpl.rcParams['axes.grid'] = True

markers = ['o','^','s','p','d']
colors = ['b','g','r','c','m','y']

from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser('W2S - UTKFace')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--test_size', type=int, default=7000, help='Test set size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--cutoff', type=float, default=0.99, help='Intrinsic dimension cutoff ratio')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
parser.add_argument('--tol_fp', type=float, default=1e-6, help='Floating point tolerance')
args = parser.parse_args('')

def get_intrinsic_dim(features, ratio=0.90, return_sval=False): 
    features = features.detach().cpu()
    sval = torch.linalg.svdvals(features)
    cumsum_sval = torch.cumsum(sval**2, dim=0)
    intrinsic_dim = torch.where(cumsum_sval >= ratio*cumsum_sval[-1])[0][0] + 1
    # intrinsic_dim = torch.where(sval**2 < 1e-3 * sval[0]**2)[0][0]
    if return_sval:
        return intrinsic_dim, sval
    return intrinsic_dim


def match_feature_dim(feaw, feas, seed):
    feaw_ = feaw.clone().detach()
    feas_ = feas.clone().detach()
    d = min(feaw_.shape[1], feas_.shape[1])
    rng = torch.Generator().manual_seed(seed)
    if feaw_.shape[1] > d:        
        feaw_ = feaw_ @ torch.randn(feaw_.shape[1], d, device=feaw_.device, generator=rng) / np.sqrt(d)
    elif feas_.shape[1] > d:
        feas_ = feas_ @ torch.randn(feas_.shape[1], d, device=feas_.device, generator=rng) / np.sqrt(d)
    return feaw_, feas_


def get_correlation_dim(feaw, feas, idimw, idims, seed=42):
    feaw_, feas_ = match_feature_dim(feaw, feas, seed)
    covar_w = feaw_.T @ feaw_ / feaw_.shape[0]
    _, evec_w = torch.linalg.eigh(covar_w)
    V_w = evec_w[:, -idimw:]
    covar_s = feas_.T @ feas_ / feas_.shape[0]
    _, evec_s = torch.linalg.eigh(covar_s)
    V_s = evec_s[:, -idims:]
    canonical_angles = torch.linalg.svdvals(V_w.T @ V_s)
    correlation_dim = torch.sum(canonical_angles**2)
    return correlation_dim


def get_correlation_dim_pad(feaw, feas, idimw, idims, seed=42):
    feaw_ = feaw.clone().detach()
    feas_ = feas.clone().detach()
    if feaw_.shape[1]!=feas_.shape[1]:
        d = max(feaw_.shape[1], feas_.shape[1])
        rng = torch.Generator().manual_seed(seed)
        if feaw_.shape[1] < d:
            feaw_ = torch.cat([feaw_, torch.randn(feaw_.shape[0], d-feaw_.shape[1], device=feaw_.device, generator=rng) / np.sqrt(d)], dim=1)
        else:
            feas_ = torch.cat([feas_, torch.randn(feas_.shape[0], d-feas_.shape[1], device=feas_.device, generator=rng) / np.sqrt(d)], dim=1)
    covar_w = feaw_.T @ feaw_ / feaw_.shape[0]
    _, evec_w = torch.linalg.eigh(covar_w)
    V_w = evec_w[:, -idimw:]
    covar_s = feas_.T @ feas_ / feas_.shape[0]
    _, evec_s = torch.linalg.eigh(covar_s)
    V_s = evec_s[:, -idims:]
    canonical_angles = torch.linalg.svdvals(V_w.T @ V_s)
    correlation_dim = torch.sum(canonical_angles**2)
    return correlation_dim

import torch
import torch.nn as nn
import torch.optim as optim

num_classes = 2
default_step = 20

# Define a simple linear classifier (Logistic Regression with Softmax)
class LinearClassifier(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)  # Linear layer

    def forward(self, x):
        return self.fc(x)  # Logits (softmax applied in loss function)

def train_classification(features, labels, ridge, lr=1.0, history_size=10, nrank=768, seed=42, step=default_step):
    model = LinearClassifier(features.shape[1], num_classes)

    
    criterion = nn.CrossEntropyLoss()  # Applies softmax internally
    optimizer = optim.LBFGS(model.parameters(), lr=lr, history_size=history_size)  # L-BFGS for fast convergence
    # Define L-BFGS closure function
    def closure():
        optimizer.zero_grad()
        logits = model(features)  # Forward pass
        loss = criterion(logits, labels)  # Compute loss
        loss.backward()  # Backpropagation
        #print(loss)
        return loss
    for _ in range(step):  # Typically 10-20 iterations is enough
        optimizer.step(closure)
    return model

# def train_ridge_regression(fea_train, lab_train, ridge, seed=42, step=20):
#     return train_classification(fea_train,lab_train,ridge)

def eval_linear_probe(prediction, features, labels):
    preds = prediction(features)
    #mse = torch.mean((preds - labels)**2).item()
    z = nn.functional.one_hot(labels,num_classes=num_classes).float()
    prob = nn.functional.softmax(preds,dim=-1)
    mse = nn.functional.mse_loss(prob, z).item()
    return mse

def linear_probe(train, test, int_dim=0, lr=1.0, history_size=10, ridge=1e-6, seed=42, step=default_step):
    fea_train, lab_train = train 
    fea_test, lab_test = test
    # if int_dim<=0:
    #     prediction = train_ridge_regression(fea_train, lab_train, ridge, seed=seed)
    # else:
    #     prediction = train_linear_regression(fea_train, lab_train, int_dim, seed=seed)
    prediction = train_classification(fea_train, lab_train, int_dim, lr=lr, history_size=history_size, seed=seed, step=step)
    mse_train = eval_linear_probe(prediction, fea_train, lab_train)
    mse_test = eval_linear_probe(prediction, fea_test, lab_test)
    return mse_test, mse_train

weak_tag = 'resnet101'
strong_tag = 'dinov2'
# features_dict = {
#     'resnet18': torch.load("./CIFAR10New/train/resnet-d_18/features.pt"), #(23705, 512)
#     'resnet34': torch.load("./CIFAR10New/train/resnet-d_34/features.pt"), #(23705, 512)
#     'resnet50': torch.load("./CIFAR10New/train/resnet_50/features.pt"), #(23705, 2048) **ResNet50 is better than ResNet101**
#     'resnet101': torch.load("./CIFAR10New/train/resnet_101/features.pt"), #(23705, 2048)
#     #'resnet152': torch.load("./precomputed/utkface_resnet152.pt"), #(23705, 2048)
#     #'clipb32': torch.load("./precomputed/utkface_clipb32.pt") #(23705, 768)
#     'dinov2': torch.load("./CIFAR10New/train/dinov2-s14/features.pt")
# }
# features_dict = {
#     'resnet18': torch.load("./precomputed/ColoredMNIST/resnet18.pt"), #(23705, 512)
#     'resnet34': torch.load("./precomputed/ColoredMNIST/resnet34.pt"), #(23705, 512)
#     'resnet50': torch.load("./precomputed/ColoredMNIST/resnet50.pt"), #(23705, 2048) **ResNet50 is better than ResNet101**
#     'resnet101': torch.load("./precomputed/ColoredMNIST/resnet101.pt"), #(23705, 2048)
#     'dinov2': torch.load("./precomputed/ColoredMNIST/dinov2-s14.pt"),
#     'vit_base': torch.load("./precomputed/ColoredMNIST/vit_base.pt")
# }
# features_w = features_dict[weak_tag]
# features_s = features_dict[strong_tag]
# labels = torch.load("./precomputed/ColoredMNIST/labels.pt")#.float() #(23705,)
# ridge = 1e-8
# n = int(4e4)

# train_index, test_index = train_test_split(range(labels.shape[0]), test_size=args.test_size, random_state=args.seed, shuffle=True)


fea_path_dict = {
    #'resnet18': "./precomputed/ColoredMNIST/resnet18.pt", #(23705, 512)
    #'resnet34': "./precomputed/ColoredMNIST/resnet34.pt", #(23705, 512)
    'resnet50': "./precomputed/ColoredMNIST/resnet_50", #(23705, 2048)
    'resnet101': "./precomputed/ColoredMNIST/resnet_101", #(23705, 2048)
    #'resnet152': "./precomputed/utkface_resnet152.pt", #(23705, 2048)
    'dinov2-s14': "./precomputed/ColoredMNIST/dinov2_s14", #(23705, 768)
    #'vit_base': "./precomputed/ColoredMNIST/vit_base.pt",
    #'mamba_base': "./precomputed/ColoredMNIST/mamba_base.pt"
}

model_label_dict = {
    #'resnet18': 'ResNet18',
    #'resnet34': 'ResNet34',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    #'resnet152': 'ResNet152',
    #'clipb32': 'CLIP-B32'
    'dinov2-s14':'DINOv2-S14',
    #'vit_base':'VIT-base'
}

class W2S_UTKFace:
    def __init__(self, args, weak_tag='resnet18', strong_tag='resnet152', ridge=1e-6, noise=0, step=default_step):
        self.args = args
        self.weak_tag = weak_tag
        self.strong_tag = strong_tag
        self.labels = torch.load(f"./precomputed/ColoredMNIST/labels_{noise}.pt")#.float()
        self.noise = noise
        self.ridge = ridge
        self.step = step
        # load features
        self.fea_w = torch.load(fea_path_dict[weak_tag]+f'_{noise}.pt')
        self.fea_s = torch.load(fea_path_dict[strong_tag]+f'_{noise}.pt')
        # intrinsic dimension + correlation dimension
        self.dw = self.get_intrinsic_dim(self.fea_w, ratio=args.cutoff)
        self.ds = self.get_intrinsic_dim(self.fea_s, ratio=args.cutoff)
        self.dsw = self.get_correlation()
        # train-test split
        idx_train, idx_test = train_test_split(range(self.labels.shape[0]), test_size=args.test_size, random_state=args.seed, shuffle=True)
        self.idx_train = idx_train
        self.idx_test = idx_test
        # artificial noise
        # if self.noise > 0:
        #     self.labels[idx_train] = self.labels[idx_train] + self.noise * torch.randn(len(idx_train))
        
    def get_intrinsic_dim(self, fea, ratio):
        return get_intrinsic_dim(fea, ratio=ratio, return_sval=False)
        
    def get_correlation(self,): 
        return get_correlation_dim(self.fea_w, self.fea_s, self.dw, self.ds)
    
    # def get_estimation_error(self,):
    #     return None
    
    def run_simulation_(self, n, N, ridge=1e-6, seed=42, verbose=True, step=5,lr=0.1,history_size=30):
        # SFT - W2S split
        idx_sft, idx_w2s = train_test_split(self.idx_train, train_size=int(n), test_size=int(N), shuffle=True, random_state=seed)
        fea_w_sft, fea_s_sft = self.fea_w[idx_sft], self.fea_s[idx_sft]
        fea_w_w2s, fea_s_w2s = self.fea_w[idx_w2s], self.fea_s[idx_w2s]
        fea_w_all, fea_s_all = torch.cat([fea_w_sft, fea_w_w2s]), torch.cat([fea_s_sft, fea_s_w2s])
        fea_w_test, fea_s_test = self.fea_w[self.idx_test], self.fea_s[self.idx_test]
        lab_sft, lab_w2s, lab_test = self.labels[idx_sft], self.labels[idx_w2s], self.labels[self.idx_test]
        lab_all = torch.cat([lab_sft, lab_w2s])

        # Weak teacher
        # fun_w = train_ridge_regression(fea_w_sft, lab_sft, ridge)
        #print('weak')
        fun_w = train_classification(fea_w_sft, lab_sft, ridge, step=default_step,lr=1)
        fun_w.eval()
        mse_w = eval_linear_probe(fun_w, fea_w_test, lab_test)
        # Strong baseline
        #print('strong')
        fun_s = train_classification(fea_s_sft, lab_sft, ridge, step=default_step,lr=1)
        fun_s.eval()
        mse_s = eval_linear_probe(fun_s, fea_s_test, lab_test)
        # Strong ceiling
        #print('ceiling')
        fun_c = train_classification(fea_s_all, lab_all, ridge, step=default_step,lr=1)
        fun_c.eval()
        mse_c = eval_linear_probe(fun_c, fea_s_test, lab_test)
        # W2S
        #print('W2S')
        #y_w2s = fun_w(fea_w_w2s)
        with torch.no_grad():
            y_w2s = nn.functional.softmax(fun_w(fea_w_w2s),dim=-1)
        #print(y_w2s.shape)
        fun_w2s = train_classification(fea_s_w2s, y_w2s,ridge, lr=lr,history_size=history_size, step=step)
        fun_w2s.eval()
        mse_w2s = eval_linear_probe(fun_w2s, fea_s_test, lab_test)
        
        pgr = (mse_w - mse_w2s) / (mse_w - mse_c)
        opr = mse_s / mse_w2s
        if verbose:
            print(f'n = {int(n)}, N = {int(N)} | excess risk: w={mse_w:.2f}, s={mse_s:.2f}, c={mse_c:.2f}, w2s={mse_w2s:.2f} | PGR={pgr:.2f}, OPR={opr:.2f}')
        return mse_w2s, mse_w, mse_s, mse_c, pgr, opr
    
    def run_simulation(self, n, N, ridge=1e-6, trials=1, verbose=True, step=default_step,lr=0.1,history_size=100):
        pgrs = []
        oprs = []
        er_w2ss = []
        er_ws = []
        er_ss = []
        er_cs = []
        for i in range(trials):
            er_w2s, er_w, er_s, er_c, pgr, opr = self.run_simulation_(n, N, ridge=ridge, seed=i, verbose=False, step=step,history_size=history_size,lr=lr)
            pgrs.append(pgr)
            oprs.append(opr)
            er_w2ss.append(er_w2s)
            er_ws.append(er_w)
            er_ss.append(er_s)
            er_cs.append(er_c)
        if verbose:
            print(f'n = {int(n)}, N = {int(N)} | PGR={pgr:.2f}, OPR={opr:.2f}, W2S={er_w2s:.2f}, Weak={er_w:.2f}, StrongBase={er_s:.2f}, StrongCeiling={er_c:.2f}')
        results = {
            'ew2s': (torch.mean(torch.tensor(er_w2ss)), torch.std(torch.tensor(er_w2ss))),
            'ew': (torch.mean(torch.tensor(er_ws)), torch.std(torch.tensor(er_ws))),
            'es': (torch.mean(torch.tensor(er_ss)), torch.std(torch.tensor(er_ss))),
            'ec': (torch.mean(torch.tensor(er_cs)), torch.std(torch.tensor(er_cs))),
            'pgr': (torch.mean(torch.tensor(pgrs)), torch.std(torch.tensor(pgrs))),
            'opr': (torch.mean(torch.tensor(oprs)), torch.std(torch.tensor(oprs))),
        }
        return results

# ------------------
# Scaling MSE
legend_dict = {
    'ew2s': 'W2S',
    'ew': 'Weak',
    'es': 'S-Baseline',
    'ec': 'S-Ceiling',
    'pgr': 'PGR',
    'opr': 'OPR',
}

# bounds_dict = {
#     'ew2s': lambda n, N, simulator: simulator.bd_exrisk_w2s(n, N),
#     'ew': lambda n, N, simulator: simulator.bd_exrisk_w(n, N),
#     'es': lambda n, N, simulator: simulator.bd_exrisk_s(n, N),
#     'ec': lambda n, N, simulator: simulator.bd_exrisk_c(n, N),
#     'pgr': lambda n, N, simulator: simulator.bd_pgr(n, N),
#     'opr': lambda n, N, simulator: simulator.bd_opr(n, N),
# }


def simulation_sample_scale(simulator, 
    n_range=[110, 400], N_range=[1000, 4000], nstart=None, Nstart=None,
    npts=20, nlines=3, trials=10
):
    N_min, N_max = N_range
    n_min, n_max = n_range
    nstart = n_min*2 if nstart is None else nstart
    Nstart = N_min*10 if Nstart is None else Nstart
    metrics = ['ew2s', 'ew', 'es', 'ec', 'pgr', 'opr']

    nn = torch.linspace(nstart, n_max, nlines, dtype=int)
    NN = torch.linspace(N_min, N_max, npts, dtype=int)
    val_scal_N = {m: (torch.zeros(nlines, npts), torch.zeros(nlines, npts)) for m in metrics}
    val_scal_N['nn'] = nn
    val_scal_N['NN'] = NN
    for i, n in enumerate(nn):
        for j, N in enumerate(NN):
            results = simulator.run_simulation(n, N, trials=trials, verbose=False)
            for key, val in results.items():
                mean, std = val
                val_scal_N[key][0][i,j] = mean
                val_scal_N[key][1][i,j] = std
            if j==0:
                print(f'n = {n}')
    print("Scaling N done")
                
    NN = torch.linspace(Nstart, N_max, nlines, dtype=int)
    nn = torch.linspace(n_min, n_max, npts, dtype=int)
    val_scal_n = {m: (torch.zeros(nlines, npts), torch.zeros(nlines, npts)) for m in metrics}
    val_scal_n['nn'] = nn
    val_scal_n['NN'] = NN
    for i, N in enumerate(NN):
        for j, n in enumerate(nn):
            results = simulator.run_simulation(n, N, trials=trials, verbose=False)
            for key, val in results.items():
                mean, std = val
                val_scal_n[key][0][i,j] = mean
                val_scal_n[key][1][i,j] = std
            if j==0:
                print(f'N = {N}')
    print("Scaling n done")
    
    return val_scal_N, val_scal_n

# weak_tag = 'resnet101'
# strong_tag = 'dinov2-s14'
# ridge = 1e-6
# results = []
# lrs = [1.0, 0.5, 0.1]
# histories = [30, 80, 130]
# steps = [3, 8, 12]
# # for lr in lrs:
# #     for history in histories:
# #         for step in steps:
# #             simulator = W2S_UTKFace(args, weak_tag=weak_tag, strong_tag=strong_tag, ridge=ridge)
# #             results.append(simulator.run_simulation(400,20000,step=step,history_size=history,lr=lr))
# simulator = W2S_UTKFace(args, weak_tag=weak_tag, strong_tag=strong_tag, ridge=ridge)

# print(f'Intrinsic dims: weak = {simulator.dw}, strong = {simulator.ds} | Correlation dim = {simulator.dsw}')
# print(f'Total training: {len(simulator.idx_train)}, Total testing: {len(simulator.idx_test)}')

# nstart = 8000
# N_min = 1500
# N_max = 50000

# Nstart = 33000
# n_min = 600
# n_max = 30000

# val_scal_N, val_scal_n = simulation_sample_scale(simulator, n_range=[n_min, n_max], N_range=[N_min, N_max], nstart=nstart, Nstart=Nstart, trials=14, nlines=1, npts=20)
# name = f'coloredmnist_{weak_tag}_{strong_tag}_ridge{ridge}_n{n_min}-{n_max}_N{N_min}-{N_max}_nstart{nstart}_Nstart{Nstart}_mse'

# # save
# os.makedirs('results',exist_ok=True)
# with open(f'./results/{name}.pkl', 'wb') as f:
#     pickle.dump((val_scal_N, val_scal_n), f)    
# print(f'Saved to ./results/{name}.pkl')

#___------------
# PGR
def simulation_sample_scale_dsw(simulators, 
    n_range=[110, 400], N_range=[1000, 4000], nstart=None, Nstart=None,
    npts=20, trials=10,
    val_scal=None, 
): # Notice: if val_scal is not None, all other arguments are ignored
    N_min, N_max = N_range
    n_min, n_max = n_range
    nstart = n_min*2 if nstart is None else nstart
    Nstart = N_min*10 if Nstart is None else Nstart
    metrics = ['ew2s', 'ew', 'es', 'ec', 'pgr', 'opr']
    NN = torch.linspace(N_min, N_max, npts, dtype=int)
    nn = torch.linspace(n_min, n_max, npts, dtype=int)

    if val_scal is None:
        val_scal_N = {m: (torch.zeros(len(simulators), npts), torch.zeros(len(simulators), npts)) for m in metrics}
        val_scal_N['n'] = nstart
        val_scal_N['NN'] = NN
        val_scal_n = {m: (torch.zeros(len(simulators), npts), torch.zeros(len(simulators), npts)) for m in metrics}
        val_scal_n['N'] = Nstart
        val_scal_n['nn'] = nn
    else:
        val_scal_N, val_scal_n = val_scal
        nstart = val_scal_N['n']
        NN = val_scal_N['NN']
        Nstart = val_scal_n['N']
        nn = val_scal_n['nn']
        for m in metrics:
            old_val_N, old_std_N = val_scal_N[m]
            add_val_N, add_std_N = torch.zeros(len(simulators), npts), torch.zeros(len(simulators), npts)
            new_val_N, new_std_N = torch.cat([add_val_N, old_val_N], dim=0), torch.cat([add_std_N, old_std_N], dim=0)
            val_scal_N[m] = (new_val_N, new_std_N)
            old_val_n, old_std_n = val_scal_n[m]
            add_val_n, add_std_n = torch.zeros(len(simulators), npts), torch.zeros(len(simulators), npts)
            new_val_n, new_std_n = torch.cat([add_val_n, old_val_n], dim=0), torch.cat([add_std_n, old_std_n], dim=0)
            val_scal_n[m] = (new_val_n, new_std_n)
        
    for i, simulator in enumerate(simulators):
        for j, N in enumerate(NN):
            results = simulator.run_simulation(nstart, N, trials=trials, verbose=False)
            for key, val in results.items():
                mean, std = val
                val_scal_N[key][0][i,j] = mean
                val_scal_N[key][1][i,j] = std
            if j==0:
                print(f'n = {nstart}')
    print("Scaling N done")
                
    for i, simulator in enumerate(simulators):
        for j, n in enumerate(nn):
            results = simulator.run_simulation(n, Nstart, trials=trials, verbose=False)
            for key, val in results.items():
                mean, std = val
                val_scal_n[key][0][i,j] = mean
                val_scal_n[key][1][i,j] = std
            if j==0:
                print(f'N = {Nstart}')
    print("Scaling n done")
    return val_scal_N, val_scal_n

ridge = 1e-6
# nstart = 1000
# nstart = 15000
# N_min = 15000
# N_max = 45000
nstart = 12000
N_min = 1500
N_max = 50000

Nstart = 33000
n_min = 600
n_max = 24000

strong_tag = 'dinov2-s14'
weak_tag = 'resnet101'
# sigmas = [0.0, 5.0, 10.0, 20.0]
sigmas = [0, 0.1, 0.2, 0.3]
simulators = [W2S_UTKFace(args, weak_tag=weak_tag, strong_tag=strong_tag, ridge=ridge, noise=sigma) for sigma in sigmas]
val_scal_N, val_scal_n = simulation_sample_scale_dsw(simulators, n_range=[n_min, n_max], N_range=[N_min, N_max], nstart=nstart, Nstart=Nstart, trials=12, npts=20)

sigma_str = '-'.join([f'{sigma:.1f}' for sigma in sigmas])
name = f'utkface_{weak_tag}_{strong_tag}_noise{sigma_str}_ridge{ridge}_n{n_min}-{n_max}_N{N_min}-{N_max}_nstart{nstart}_Nstart{Nstart}'
#name
# with open(f'./results/{name}.pkl', 'rb') as f:
#     val_scal_N, val_scal_n = pickle.load(f)

with open(f'./results/{name}.pkl', 'wb') as f:
    pickle.dump((val_scal_N, val_scal_n), f)    
print(f'Saved to ./results/{name}.pkl')