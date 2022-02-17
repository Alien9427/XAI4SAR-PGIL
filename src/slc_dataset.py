import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob, torch
from torch.utils.data.sampler import BatchSampler
import cv2
import random

def get_loc(idx, col_num):
    i = int(idx / col_num)
    j = int(idx % col_num)
    return i, j


class Comp_Ice_Dataset(Dataset):
    def __init__(self, data_txt, cate_txt, data_root, transform=None):
        self.data_root = data_root
        self.data = pd.read_csv(data_txt)
        self.cate = pd.read_csv(cate_txt)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ idx 是list格式 """
        img = np.load(self.data_root + self.data.loc[idx]['path'] + '.npy')
        scat_path = self.data_root + self.data.loc[idx]['path'] + '_scat.npy'
        catename = self.data.loc[idx]['catename']
        label = int(self.cate.loc[self.cate['catename'] == catename]['label'])

        sample = {
            'img': img,
            'scat_path': scat_path,
            'catename': catename,
            'label': label
        }

        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample

class Comp_Ice_Dataset_un(Dataset):
    def __init__(self, data_txt, data_root, transform=None):
        self.data_root = data_root
        self.data = pd.read_csv(data_txt)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.load(self.data_root + self.data.loc[idx]['path'] + '.npy')
        scat_path = self.data_root + self.data.loc[idx]['path'] + '_scat.npy'

        sample = {
            'img': img,
            'scat_path': scat_path,
        }

        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size