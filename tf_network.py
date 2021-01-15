import tensorflow as tf
import tensorflow.nn as nn
from tensorflow.contrib.layers import flatten


class neuralNetwork:

    def __init__(self):
        '''
        Define some basic parameters here
        '''

        pass

    def Net(self, input):
        '''
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        '''
        # Define the parameters
        conv1_W = self.init_weight([5, 5, 1, 6])
        con1_b = self.init_bias(6)

        conv2_W = self.init_weight([5, 5, 6, 16])
        con2_b = self.init_bias(16)

        fc1_W = self.init_weight([16 * 5 * 5, 120])
        fc1_b = self.init_bias(120)

        fc2_W = self.init_weight([120, 84])
        fc2_b = self.init_bias(84)

        fc3_W = self.init_weight([84, 10])
        fc3_b = self.init_bias(10)

        '''Define the architecture of the network'''
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        x = nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + con1_b
        x = nn.relu(x)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='VALID')  # [batch, height, width, channels]

        # Layer 2: Convolutional. Output = 10x10x16.
        x = nn.conv2d(x, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + con2_b
        x = nn.relu(x)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='VALID')  # [batch, height, width, channels]

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        x = flatten(x)
        x = tf.matmul(x, fc1_W) + fc1_b
        x = nn.relu(x)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        x = tf.matmul(x, fc2_W) + fc2_b
        x = nn.relu(x)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        x = tf.matmul(x, fc3_W) + fc3_b
        # x = nn.relu(x)
        logits = x  # logits.shape = (batch_size, 10)

        return logits

    def forward(self, input):
        '''
        Forward the network
        '''
        return self.Net(input)

    def init_weight(self, shape):
        '''
        Init weight parameter.
        '''
        w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        return tf.Variable(w)

    def init_bias(self, shape):
        '''
        Init bias parameter.
        '''
        b = tf.zeros(shape)
        return tf.Variable(b)
