# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import ImageFile
#from torchvision import transforms
#import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Dataset
#from torchvision.datasets import MNIST, ImageFolder
#from torchvision.transforms.functional import rotate
#from sklearn.model_selection import StratifiedShuffleSplit
#import numpy as np

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset
#from domainbed.lib.misc import _SplitDataset#, SimSiamTransform
from transformers import AutoTokenizer
import requests
import pandas as pd
import zipfile
#import random
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "SST2",
 
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class IndexedDataset(Dataset):
    def __init__(self, dataset,label_only = False,include_color=False):
        self.dataset = dataset
        self.label_only = label_only
        self.include_color = include_color
        try:
            self.classes = dataset.classes
        except:
            pass
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.include_color:
            image,label,color = self.dataset[index]
            return image, label, color, index
        else:
            image, label = self.dataset[index]
            if self.label_only:
                return 0, label, index
            return image, label, index

class SST2(MultipleDomainDataset):
    """ SST-2 dataset with three environments: teacher, student, and test. """

    ENVIRONMENTS = ["teacher", "student", "test"]

    def __init__(self, path='SST-2/SST-2', tokenizer_name="bert-base-uncased", split_ratio=[0.4, 0.4, 0.2], max_length=300, seed=42):
        """
        Initializes SST-2 dataset, downloads it if necessary, tokenizes it, and splits into environments.

        Args:
            tokenizer_name (str): Name of the tokenizer from Hugging Face.
            split_ratio (list): Fraction of data allocated to ['teacher', 'student', 'test'].
            max_length (int): Maximum tokenized sequence length.
            seed (int): Random seed for reproducibility.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.dataset_path = path
        self.data_url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

        # Download and extract dataset if necessary
        #self._download_and_extract()

        # Load and shuffle dataset
        df = pd.read_csv(os.path.join(self.dataset_path, "train.tsv"), delimiter="\t").sample(frac=1, random_state=seed)

        # Split dataset and tokenize in one step
        self._split_and_tokenize(df, split_ratio)

        self.input_shape = None
        self.num_classes = 2

    def _download_and_extract(self):
        """ Downloads and extracts SST-2 dataset if not already available. """
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        zip_path = os.path.join(self.dataset_path, "SST-2.zip")

        if not os.path.exists(zip_path):
            print("Downloading SST-2 dataset...")
            response = requests.get(self.data_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 KB

            with open(zip_path, "wb") as file, tqdm(
                desc="Downloading", total=total_size, unit="B", unit_scale=True
            ) as progress:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(len(data))

            print("Download complete.")

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dataset_path)

        print("Extraction complete.")

    def _split_and_tokenize(self, df, split_ratio):
        """ Splits the dataset into teacher, student, and test environments and pre-tokenizes it. """
        total_samples = len(df)
        teacher_size = int(split_ratio[0])
        student_size = int(split_ratio[1])
        test_size = int(split_ratio[2])
        if sum(split_ratio) > total_samples:
            raise ValueError("Dataset total sample number exceeded")
        if split_ratio[0] <= 1:
            teacher_size = int(split_ratio[0] * total_samples)
        if split_ratio[1] <= 1:
            student_size = int(split_ratio[1] * total_samples)
        if split_ratio[2] <= 1:
            test_size = int(split_ratio[2] * total_samples)

        splits = {
            "teacher": df.iloc[:teacher_size],
            "student": df.iloc[-test_size-student_size:-test_size],
            "test": df.iloc[-test_size:]
        }

        # Pre-tokenize and store as PyTorch datasets
        # self.datasets = {env: SST2TorchDataset(splits[env], self.tokenizer, self.max_length) for env in splits}
        self.datasets = [SST2TorchDataset(splits[env], self.tokenizer, self.max_length) for env in splits]

    # def get_environment(self, env_name):
    #     """ Returns the pre-tokenized PyTorch dataset for the specified environment. """
    #     if env_name not in self.datasets:
    #         raise ValueError(f"Environment {env_name} not found. Choose from {self.ENVIRONMENTS}.")
    #     return self.datasets[env_name]

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
            'idx': idx
        }

