#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: cifar_10.py
Date: 2021/1/15 11:21
LastEditors: LI Jinjie
LastEditTime: 2021/1/15 11:21
Description: read data of cifar-10
'''

import numpy as np
import pickle
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self):
        self.images = None
        self.labels = None


class Cifar_10:
    '''
    read data from Cifar_10 and return it with the ideal form
    '''

    def __init__(self):
        self.train = DataSet()  # 45000
        self.test = DataSet()  # 10000
        self.validation = DataSet()  # 5000

    def read_data_sets(self, f_path):
        with open(f_path + 'data_batch_1', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.array(dict[b'data'])
            self.train.labels = np.array(dict[b'labels'])[:, np.newaxis]

        with open(f_path + 'data_batch_2', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(self.train.images, np.array(dict[b'data']), axis=0)
            self.train.labels = np.append(self.train.labels, np.array(dict[b'labels'])[:, np.newaxis], axis=0)

        with open(f_path + 'data_batch_3', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(self.train.images, np.array(dict[b'data']), axis=0)
            self.train.labels = np.append(self.train.labels, np.array(dict[b'labels'])[:, np.newaxis], axis=0)

        with open(f_path + 'data_batch_4', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(self.train.images, np.array(dict[b'data']), axis=0)
            self.train.labels = np.append(self.train.labels, np.array(dict[b'labels'])[:, np.newaxis], axis=0)

        with open(f_path + 'data_batch_5', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(self.train.images, np.array(dict[b'data']), axis=0)
            self.train.labels = np.append(self.train.labels, np.array(dict[b'labels'])[:, np.newaxis], axis=0)

        with open(f_path + 'test_batch', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.test.images = np.array(dict[b'data'])
            self.test.labels = np.array(dict[b'labels'])[:, np.newaxis]

        data = np.append(self.train.images, self.train.labels, axis=1)
        train_set, validation_set = train_test_split(data, test_size=0.1, random_state=24)
        self.train.images = train_set[:, :-1]
        self.train.labels = train_set[:, -1]

        self.validation.images = validation_set[:, :-1]
        self.validation.labels = validation_set[:, -1]


if __name__ == "__main__":
    cifar_10 = Cifar_10()
    f_path = "data/cifar-10-batches-py/"
    cifar_10.read_data_sets(f_path)
    pass
