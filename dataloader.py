import os
from glob import glob

import nibabel as nib
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils.module import normalize_image
from utils.dataset import PairedDataset

class DataLoaderGenerator():
    def __init__(self, main_args):
        self.test_size = 0.3
        self.random_state = 41
        self.main_args = main_args

    def generate_random_train_and_valid_subsets_paths(self):
        root_path = self.main_args.root_path
        patient_dirs = os.listdir(root_path)

        train_dirs, val_dirs = train_test_split(patient_dirs, test_size=self.test_size, random_state=self.random_state)

        return train_dirs, val_dirs

    def generate(self):
        train_dirs, val_dirs = self.generate_random_train_and_valid_subsets_paths()

        print(f"Number of training samples: {len(train_dirs)}")
        print(f"Number of validation samples: {len(val_dirs)}")

        train_dataset = PairedDataset(train_dirs, self.main_args.root_path, augment=False)
        val_dataset = PairedDataset(val_dirs, self.main_args.root_path, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.main_args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.main_args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader

class DataLoaderGeneratorTest():
    def __init__(self, main_args):

        self.main_args = main_args

    def generate_test_paths(self):
        root_path = self.main_args.test_path
        patient_dirs = os.listdir(root_path)
        return patient_dirs

    def generate(self):
        test_dirs = self.generate_test_paths()

        print(f"Number of test samples: {len(test_dirs)}")

        test_dataset = PairedDataset(test_dirs, self.main_args.test_path, augment=False)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.main_args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
        return test_loader