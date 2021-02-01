#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: data_object.py
Date: 2021/1/15 11:21
LastEditors: LI Jinjie
LastEditTime: 2021/1/15 11:21
Description: process raw data and return the appropriate format.
'''

import numpy as np
import collections
from sklearn.utils import shuffle
import cv2
import pickle
from PIL import Image
import PIL
import matplotlib.pyplot as plt


class DATA_OBJECT(object):
    def __init__(self,
                 images,
                 labels,
                 num_classes=0,
                 one_hot=False,
                 dtype=np.float32,
                 reshape=False):
        """
        Data object construction.
        Input parameter:
            - images: The images of size [num_samples, rows, columns, depth].
            - labels: The labels of size [num_samples,]
            - num_classes: The number of classes in case one_hot labeling is desired.
            - one_hot=False: Turn the labels into one_hot format.
            - dtype=np.float32: The data type.
            - reshape=False: Reshape in case the feature vector extraction is desired.
        """
        # Define the date type.
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_samples = images.shape[0]

        # [num_examples, rows, columns, depth] -> [num_examples, rows*columns]
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])

        # Conver to float if necessary
        if dtype == np.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(dtype)
            images = np.multiply(images, 1.0 / 255.0)

        # shuffle images and labels
        images, labels = shuffle(images, labels)

        self._images = images
        self._labels = labels

        # If the one_hot flag is true, then the one_hot labeling supersedes the normal labeling.
        if one_hot:
            # If the one_hot labeling is desired, number of classes must be defined as one of the arguments of DATA_OBJECT class!
            assert num_classes != 0, (
                'You must specify the num_classes in the DATA_OBJECT for one_hot label construction!')

            # Define the indexes.
            index = np.arange(
                self._num_samples) * num_classes  # np.arange(5)=[0,1,2,3,4]    np.arange(5)*3=[0,3,6,9,12]
            one_hot_labels = np.zeros((self._num_samples, num_classes))
            one_hot_labels.flat[
                index + labels.ravel()] = 1  # row index + labels number = ont_hot  shape = (num_samples, num_classes)
            self._labels = one_hot_labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_samples(self):
        return self._num_samples


def provide_data(cifar_10):
    """
    This function provide data object with desired shape.
    The attribute of data object:
        - train
        - validation
        - test
    The sub attributs of the data object attributes:
        -images
        -labels

    :param cifar_10: The downloaded cifar-10 dataset
    :return: data: The data object.
                   ex: data.train.images return the images of the dataset object in the training set!
    """
    ################################################
    ########## Get the images and labels############
    ################################################

    IMAGE_SIZE = 224

    # The ?_images(? can be train, validation or test) must have the format of [num_samples, rows, columns, depth] after extraction from data.
    # The ?_labels(? can be train, validation or test) must have the format of [num_samples,] after extraction from data.
    # from batch, channels, height, width to batch, height, width, channels; RGB
    train_images_org = cifar_10.train.images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    train_images = train_images_org
    train_labels = cifar_10.train.labels

    validation_images_org = cifar_10.validation.images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    validation_images = validation_images_org
    validation_labels = cifar_10.validation.labels

    test_images_org = cifar_10.test.images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_images = test_images_org
    test_labels = cifar_10.test.labels

    # Create separate objects for train, validation & test.
    train = DATA_OBJECT(train_images, train_labels, num_classes=10, one_hot=True, dtype=np.float32, reshape=False)
    validation = DATA_OBJECT(validation_images, validation_labels, num_classes=10, one_hot=True, dtype=np.float32,
                             reshape=False)
    test = DATA_OBJECT(test_images, test_labels, num_classes=10, one_hot=True, dtype=np.float32, reshape=False)

    # Create the whole data object
    DataSetObject = collections.namedtuple('DataSetObject', ['train', 'validation',
                                                             'test'])  # data = DataSetObject(1,2,3)  data.train = 1   data.test = 3
    data = DataSetObject(train=train, validation=validation, test=test)

    return data


def preprocess(images_org, IMAGE_SIZE):
    # resize the image
    images = np.zeros([images_org.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3], dtype=images_org.dtype)
    for i in range(images_org.shape[0]):
        images[i] = cv2.resize(images_org[i], dsize=(IMAGE_SIZE, IMAGE_SIZE),
                               interpolation=cv2.INTER_LINEAR)
    # i = 0
    # while (i < 1000):
    #     plt.imshow(images[i])
    #     plt.show()
    #     i += 1

    return images


if __name__ == "__main__":
    img = np.fromfile("D:\\ForGithub\\AlexNet_TF\\bin_data_from_ubuntu\\257.bin", dtype=np.float32)
    img = np.reshape(img, [224, 224, 3])
    plt.imshow(img)
    plt.show()
    pass

# if __name__ == "__main__":
#     IMAGE_SIZE = 224
#
#     f_path = "./data/cifar-10-batches-py/test_batch"
#     with open(f_path, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#         images = np.array(dict[b'data'], dtype=np.uint8)
#         labels = np.array(dict[b'labels'])[:, np.newaxis]
#
#         np.savetxt('labels.txt', labels, fmt='%d')
#
#     images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
#
#     for i in range(images.shape[0]):
#         # images[i] = cv2.resize(images_org[i], dsize=(IMAGE_SIZE, IMAGE_SIZE),
#         #                        interpolation=cv2.INTER_LINEAR)
#         im = Image.fromarray(np.uint8(images[i]))
#         im_large = im.resize((IMAGE_SIZE, IMAGE_SIZE), resample=PIL.Image.BILINEAR)
#         # img_array = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.float32)
#         img_array = np.multiply(np.array(im_large), 1.0 / 255.0).astype(np.float32)
#
#         plt.imshow(img_array)
#         plt.show()
#
#         print(img_array.shape)
#
#         # make target
#         outputName = ("./data_test/{}.bin".format(i))
#         img_array.tofile(outputName)
#         print("image {} has completed.".format(outputName))
