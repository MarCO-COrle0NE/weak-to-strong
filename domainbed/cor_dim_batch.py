import os
import argparse

import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd

class SST2TorchDataset(Dataset):
    """ PyTorch Dataset Wrapper for SST-2 with Pre-tokenized Inputs """
    def __init__(self, df, tokenizer, max_length):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for _, row in df.iterrows():
            encoding = tokenizer(
                row["sentence"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )
            self.input_ids.append(encoding["input_ids"].squeeze(0))
            self.attention_masks.append(encoding["attention_mask"].squeeze(0))
            self.labels.append(torch.tensor(row["label"], dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            # 'idx': idx
        }


def get_gradients(model, dataset, tokenizer, d_s, device='cuda', transform_step=4, selected_indices=None):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # Batch size of 1 for simplicity
    all_gradients = []
    for step, batch in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # target = probs[:, 0].sum()  # Just a simple target to compute gradients. You may change this.
        # target.backward()
        # Convert labels to one-hot encoding
        num_classes = logits.size(-1)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        one_hot_labels = one_hot_labels.to(device).type_as(probs)

        # Compute squared loss between probabilities and one-hot labels
        loss = torch.nn.functional.mse_loss(probs, one_hot_labels,reduction='sum')  # Squared loss
        loss.backward()  # Backpropagate to compute gradients

        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.reshape(-1))
        gradients = torch.cat(gradients)
        if selected_indices is not None:
            gradients = gradients[selected_indices]
        all_gradients.append(gradients)
        if step == transform_step:
            print(gradients.shape)
            x = torch.stack(all_gradients)
            print(x.shape)
            result = to_intrinsic(x, d_s, device=device)
            all_gradients = []
        elif step > 0 and step % transform_step == 0:
            result += to_intrinsic(torch.stack(all_gradients), d_s, device=device)
            all_gradients = []
    return result  # D_w by d


def to_intrinsic(F, d_s, device='cuda'):
    N = F.size(0)
    G = torch.randn(N, d_s, device=device).half()  # Convert G to float16
    FG = torch.matmul(F.T, G) / torch.sqrt(torch.tensor(d_s, dtype=torch.float16, device=device))  # Use float16
    return FG


def orthonormalize_gradients(F, d_s, device='cuda'):
    FG = to_intrinsic(F, d_s, device=device)
    Q = ortho(FG)
    return Q


def get_intrinsic_matrix(model_name, train_data, d_s, device='cuda', transform_step=4, D_w=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.half()  # Convert model to float16
    model.to(device)
    D = sum(p.numel() for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = SST2TorchDataset(train_data, tokenizer, max_length=300)
    if D_w:
        selected_indices = torch.randperm(D)[:D_w]
    else:
        selected_indices = None
    F = get_gradients(model, dataset, tokenizer, d_s, device=device, transform_step=transform_step, selected_indices=selected_indices)
    return ortho(F), D


def ortho(C):
    # Convert C to float32 for QR decomposition
    C_float32 = C.float()
    Q, _ = torch.linalg.qr(C_float32)
    # Convert Q back to float16 if needed
    Q = Q.half()
    return Q


def calculate_norm(V_s, V_w, D_s, D_w):
    """
    ||V_s^T U V_w||^2_F
    """
    result = torch.matmul(V_s.T, V_w)
    norm_squared = torch.norm(result, p='fro') ** 2
    return norm_squared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_name', type=str, default='train.tsv')
    parser.add_argument('--model_name_s', type=str, default="bert-base-uncased")
    parser.add_argument('--model_name_w', type=str, default="prajjwal1/bert-mini")
    parser.add_argument('--d_s', type=int, default=1000)
    parser.add_argument('--d_w', type=int, default=4000)
    parser.add_argument('--transform_freq', type=int, default=4)
    parser.add_argument('--common_D',type=int, default=None)
    parser.add_argument('--multi_device', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # if args.deterministic_only:
    #     torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    data_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(data_dir, "train.tsv"), delimiter="\t")

    if args.multi_device:
        device_w = torch.device("cuda:0")
        device_s = torch.device("cuda:1")

        model_name = args.model_name_w
        V_w, D_w = get_intrinsic_matrix(model_name, train_data, args.d_w, device=device_w, transform_step=args.transform_freq, D_w=args.common_D)
        print("Orthonormalized matrix shape (V_w):", V_w.shape)

        if args.common_D:
            D_w = args.common_D
        model_name = args.model_name_s
        V_s, D_s = get_intrinsic_matrix(model_name, train_data, args.d_s, device=device_s, transform_step=args.transform_freq, D_w=D_w)
        print("Orthonormalized matrix shape (V_s):", V_s.shape)

        print(D_s, D_w)

        torch.cuda.synchronize(device_w)
        torch.cuda.synchronize(device_s)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = args.model_name_w
        V_w, D_w = get_intrinsic_matrix(model_name, train_data, args.d_w, device=device, transform_step=args.transform_freq, D_w=args.common_D)
        print("Orthonormalized matrix shape (V_w):", V_w.shape)

        if args.common_D:
            D_w = args.common_D
        model_name = args.model_name_s
        V_s, D_s = get_intrinsic_matrix(model_name, train_data, args.d_s, device=device, transform_step=args.transform_freq, D_w=D_w)
        print("Orthonormalized matrix shape (V_s):", V_s.shape)

        print(D_s, D_w)

    d_sw = calculate_norm(V_s, V_w, D_s, D_w)
    print('Correlation dimension:', d_sw)