
import os
import argparse

import torch
from torch.utils.data import DataLoader,Dataset
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
            # "labels": self.labels[idx],
            # 'idx': idx
        }


def get_gradients(model, dataset, tokenizer, d_s, device='cuda', transform_step=4):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size of 1 for simplicity
    all_gradients = []
    for step,batch in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}# if k in tokenizer.model_input_names}
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits,dim=-1)
        target = probs[:, 0].sum()  # Just a simple target to compute gradients. You may change this.
        target.backward()

        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.reshape(-1))
        gradients = torch.cat(gradients)
        all_gradients.append(gradients)
        if step == transform_step:
            print(gradients.shape)
            x = torch.stack(all_gradients)
            print(x.shape)
            result = to_intrinsic(x,d_s,device=device)
            all_gradients = []
        elif step > 0 and step % transform_step == 0:
            result += to_intrinsic(torch.stack(all_gradients),d_s,device=device)
            all_gradients = []
    return result # D by d

def to_intrinsic(F,d_s,device='cuda'):
    N = F.size(0)
    G = torch.randn(N, d_s, device=device)
    FG = torch.matmul(F.T, G) / torch.sqrt(torch.tensor(d_s, dtype=torch.float32, device=device))
    return FG

def orthonormalize_gradients(F, d_s, device='cuda'):
    FG = to_intrinsic(F,d_s,device=device)
    Q = ortho(FG)
    return Q

def get_intrinsic_matrix(model_name,train_data,d_s,device='cuda',transform_step=4):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    D = sum(p.numel() for p in model.parameters())
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #encoded_inputs = tokenizer(train_data['sentence'].tolist(), padding=True, truncation=True, return_tensors='pt')
    #dataset = torch.utils.data.TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
    dataset = SST2TorchDataset(train_data,tokenizer,max_length=300)
    F = get_gradients(model, dataset, tokenizer, d_s, device=device, transform_step=transform_step)
    return ortho(F), D

def ortho(C):
    Q, _ = torch.linalg.qr(C)
    return Q

def calculate_norm(V_s, V_w, D_s, D_w):
    """
    ||V_s^T U V_w||^2_F
    """
    D_1 = V_s.size(0)
    D_2 = V_w.size(0)
    print(D_1,D_2)
    C = torch.randn(D_1, D_2)
    U = ortho(C)
    # V_s^T U V_w
    result = torch.matmul(torch.matmul(V_s.T, U), V_w)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(data_dir, "train.tsv"), delimiter="\t")

    model_name = args.model_name_s
    V_s, D_s = get_intrinsic_matrix(model_name,train_data,args.d_s,device=device,transform_step=args.transform_freq)
    print("Orthonormalized matrix shape (V_s):", V_s.shape)

    model_name = args.model_name_w
    V_w, D_w = get_intrinsic_matrix(model_name,train_data,args.d_w,device=device,transform_step=args.transform_freq)
    print("Orthonormalized matrix shape (V_w):", V_w.shape)

    print(D_s,D_w)

    d_sw = calculate_norm(V_s,V_w,D_s,D_w)
    print('Correlation dimension:',d_sw)



