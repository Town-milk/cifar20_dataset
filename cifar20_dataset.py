import torch
import numpy as np
from torch.utils.data import Dataset

class CIFAR20_Dataset(Dataset):
    def __init__(self, dataRootPath="./cifar20_dataset/", train=True, numData=100):
        if train:
            self.imgs = np.load(dataRootPath + "cifar20_data%d_train_imgs.npy"%numData)
            self.labels = np.load(dataRootPath + "cifar20_data%d_train_labels.npy"%numData)
        else:
            self.imgs = np.load(dataRootPath + "cifar20_test_imgs.npy")
            self.labels = np.load(dataRootPath + "cifar20_test_labels.npy")

        self.imgs = np.transpose(self.imgs.data, (0, 3, 1, 2))
        self.imgs = self.imgs / 255.
        self.imgs = torch.tensor(self.imgs.astype(np.float32))
        self.labels = torch.tensor(self.labels)


    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)