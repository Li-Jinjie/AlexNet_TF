#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: tf_test_model.py
Date: 2021/1/15 11:25
LastEditors: LI Jinjie
LastEditTime: 2021/1/15 11:21
Description: Test the .pb model saved in path:./model .
'''

import tensorflow as tf
from tensorflow.python.platform import gfile
from data_object import provide_data
from cifar_10 import Cifar_10
from data_object import preprocess
from tqdm import tqdm


def test_model(batch_size, dataset):
    '''
    Test model function.
    Return: accuracy and loss of the dataset.
    '''
    num_samples = dataset.num_samples   # the number of sample in dataset
    # print(num_samples)
    data_images = dataset.images        # get images data
    data_labels = dataset.labels        # get labels data

    total_accuracy = 0
    total_loss = 0
    for offset in tqdm(range(0, num_samples, batch_size)):
        # "offset" is the start position of the index, "end" is the end position of the index.
        end = offset + batch_size
        batch_images, batch_labels = data_images[offset:end], data_labels[offset:end]   # get images and labels according to the batch number
        batch_images = preprocess(batch_images, 224)  # resize the images
        total_accuracy += sess.run(accuracy_operation, feed_dict={input_images: batch_images, images_labels: batch_labels})
        total_loss += sess.run(loss_operation, feed_dict={input_images: batch_images, images_labels: batch_labels})
    
    return (total_accuracy * batch_size / num_samples, total_loss * batch_size / num_samples)


if __name__ == "__main__":
    
    batch_size = 512   # batch size, you can change the value of it

    # create session
    with tf.Session() as sess:
        # open the model and get graph
        with gfile.FastGFile('./model/AlexNet_model_0675.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='') # import graph
        
        # prepare training dataset and test dataset
        # train: 55000, test: 10000, validation: 5000
        cifar_10 = Cifar_10()
        f_path = "data/cifar-10-batches-py/"
        cifar_10.read_data_sets(f_path)
        # load cifar-10 dataset
        data = provide_data(cifar_10)

        # init session
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # get input tensor
        input_images = sess.graph.get_tensor_by_name('input:0')             # input parameter: images
        images_labels = sess.graph.get_tensor_by_name('labels:0')           # input parameter: labels
        accuracy_operation = sess.graph.get_tensor_by_name('accuracy:0')    # output parameter: accuracy
        loss_operation = sess.graph.get_tensor_by_name('loss_op:0')            # output parameter: loss
        
        # get accuracy and loss
        test_accuracy, loss = test_model(batch_size, data.test)

        print("Test Loss = {:.5f}, Test Accuracy = {:.5f}\n".format(loss, test_accuracy))
