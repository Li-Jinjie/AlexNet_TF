#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: cifar_10.py
Date: 2021/1/15 11:21
LastEditors: LI Jinjie
LastEditTime: 2021/1/15 11:21
Description: file content
'''

import numpy as np
import pickle


class DataSet:
    def __init__(self):
        self.images = None
        self.labels = None


class Cifar_10:
    '''
    read data from Cifar_10 and return it with the ideal form
    '''

    def __init__(self):
        self.train = DataSet()
        self.test = DataSet()
        self.validation = DataSet()

    def read_data_sets(self, f_path):
        with open(f_path + 'data_batch_1', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = dict[b'data']
            self.train.labels = dict[b'labels']

        with open(f_path + 'data_batch_2', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(dict[b'data'], axis=0)
            self.train.labels = np.append(dict[b'labels'], axis=0)

        with open(f_path + 'data_batch_3', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(dict[b'data'], axis=0)
            self.train.labels = np.append(dict[b'labels'], axis=0)

        with open(f_path + 'data_batch_4', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(dict[b'data'], axis=0)
            self.train.labels = np.append(dict[b'labels'], axis=0)

        with open(f_path + 'data_batch_5', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.train.images = np.append(dict[b'data'], axis=0)
            self.train.labels = np.append(dict[b'labels'], axis=0)

        with open(f_path + 'test_batch', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.test.images = dict[b'data']
            self.test.labels = dict[b'labels']

        # with open(file, 'rb') as fo:
        #     dict = pickle.load(fo, encoding='bytes')
        #     self.test.images = dict[b'data']
        #     self.test.labels = dict[b'labels']
        #     pass


if __name__ == "__main__":
    cifar_10 = Cifar_10()
    f_path = "data/cifar-10-batches-py/"
    cifar_10.read_data_sets(f_path)
    pass
