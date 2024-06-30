import glob
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.Custom_Datasets import Custom_dataset
from datasets.transform import *
from tqdm import tqdm
import pickle
from torchvision.transforms import v2
from scipy import signal
import math


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
        Reshape(),
        v2.RandomHorizontalFlip(p=0.5),
        Retype(),
    ]),
        'val': Compose([
        Reshape(),
        Retype(),
    ])
    }
    return transforms[dataset_type]


def train_test_split_order(data_pd, test_size=0.8, num_classes=2):
    train_pd = pd.DataFrame(columns=('path_list', 'label'))
    val_pd = pd.DataFrame(columns=('path_list', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd._append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['path_list', 'label']], ignore_index=True)
        val_pd = val_pd._append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['path_list', 'label']], ignore_index=True)
    return train_pd,val_pd


class two_cati_CWRUSTFT(object):
    num_classes = 2
    inputchannel = 1
    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):

        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            data_dir = self.data_dir
            data_dir1 = os.path.join(data_dir, r'fault\CWRU_12K')
            path_list1 = glob.glob(os.path.join(data_dir1, '*'), recursive=True)
            label_list1 = np.ones(len(path_list1))
            data_dir2 = os.path.join(data_dir, r'fault\CWRU_48K')
            path_list2 = glob.glob(os.path.join(data_dir2, '*'), recursive=True)
            label_list2 = np.ones(len(path_list2))
            data_dir3 = os.path.join(data_dir, r'health\CWRU_48K')
            path_list3 = glob.glob(os.path.join(data_dir3, '*'), recursive=True)
            label_list3 = np.zeros(len(path_list3))

            path_list = path_list1 + path_list2 + path_list3
            label_list = np.concatenate((label_list1, label_list2, label_list3)).tolist()

        if test:
            test_dataset = Custom_dataset(list_data=path_list, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"path_list": path_list, "label": label_list})
            train_pd, val_pd = train_test_split_order(data_pd, test_size=0.2, num_classes=2)
            train_dataset = Custom_dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = Custom_dataset(list_data=val_pd)
            return train_dataset, val_dataset





