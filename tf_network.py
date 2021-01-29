import tensorflow as tf
import tensorflow.nn as nn
from tensorflow.contrib.layers import flatten


class neuralNetwork:

    def __init__(self):
        '''
        Define some basic parameters here
        '''

        pass

    def Net(self, input, prob=0.5):
        '''
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        '''
        # Define the parameters
        conv1_W = self.init_weight([11, 11, 3, 96])
        conv1_b = self.init_bias(96)

        conv2_W = self.init_weight([5, 5, 96, 256])
        conv2_b = self.init_bias(256)

        conv3_W = self.init_weight([3, 3, 256, 384])
        conv3_b = self.init_bias(384)

        conv4_W = self.init_weight([3, 3, 384, 384])
        conv4_b = self.init_bias(384)

        conv5_W = self.init_weight([3, 3, 384, 256])
        conv5_b = self.init_bias(256)

        fc1_W = self.init_weight([5 * 5 * 256, 4096])
        fc1_b = self.init_bias(4096)

        fc2_W = self.init_weight([4096, 4096])
        fc2_b = self.init_bias(4096)

        fc3_W = self.init_weight([4096, 10])
        fc3_b = self.init_bias(10)

        '''Define the architecture of the network'''
        # N=(W-F+2P)/S+1
        # Layer 1: Convolutional. Input = 224x224x3. Output = 54x54x96.
        x = nn.conv2d(input, conv1_W, strides=[1, 4, 4, 1], padding=[[0, 0], [1, 1], [1, 1], [0, 0]]) + conv1_b
        x = nn.relu(x, name="conv_layer_01/relu")
        # LRN. depth_radius = n/2 = 3, bias = k = 2. Following the paper.
        x = nn.lrn(x, 3, bias=2.0, alpha=1e-4, beta=0.75, name="conv_layer_01/lrn1")
        # Pooling. Input = 54x54x96. Output = 26x26x96.
        x = nn.max_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='VALID', name="conv_layer_01/pooling")  # [batch, height, width, channels]

        # Layer 2: Convolutional. Input = 26x26x96. Output = 26x26x256.
        x = nn.conv2d(x, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        x = nn.relu(x, name="conv_layer_02/relu")
        # LRN. depth_radius = n/2 = 3, bias = k = 2. Following the paper.
        x = nn.lrn(x, 3, bias=2.0, alpha=1e-4, beta=0.75, name="conv_layer_01/lrn1")
        # Pooling. Input = 26x26x256. Output = 12x12x256.
        x = nn.max_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="conv_layer_02/pooling")

        # Layer 3: Convolutional. Input = 12x12x256. Output = 12x12x384.
        x = nn.conv2d(x, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        x = nn.relu(x, name="conv_layer_03/relu")

        # Layer 4: Convolutional. Input = 12x12x384. Output = 12x12x384.
        x = nn.conv2d(x, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
        x = nn.relu(x, name="conv_layer_04/relu")

        # Layer 5: Convolutional. Input = 12x12x384. Output = 12x12x256.
        x = nn.conv2d(x, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
        x = nn.relu(x, name="conv_layer_05/relu")
        # Pooling. Input = 12x12x256. Output = 5x5x256.
        x = nn.max_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="conv_layer_05/pooling")

        # Layer 6: Fully Connected. Input = 5x5x256=6400. Output = 4096.
        x = flatten(x)
        x = tf.matmul(x, fc1_W) + fc1_b
        x = nn.relu(x, name="full_layer_01/relu")
        # Dropout
        x = nn.dropout(x, rate=prob)

        # Layer 7: Fully Connected. Input = 4096. Output = 4096.
        x = tf.matmul(x, fc2_W) + fc2_b
        x = nn.relu(x, name="full_layer_02/relu")
        # Dropout
        x = nn.dropout(x, rate=prob)

        # # Layer 8: Fully Connected. Input = 4096. Output = 10.
        x = tf.add(tf.matmul(x, fc3_W), fc3_b, name="full_layer_03/linear")

        logits = x  # logits.shape = (batch_size, 10)

        return logits

    def forward(self, input, prob):
        '''
        Forward the network
        '''
        return self.Net(input, prob)

    def init_weight(self, shape):
        '''
        Init weight parameter.
        '''
        w = tf.random.truncated_normal(shape=shape, mean=0, stddev=0.01)  # 0.1
        return tf.Variable(w)

    def init_bias(self, shape):
        '''
        Init bias parameter.
        '''
        # b = tf.zeros(shape)

        # This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs.
        b = tf.ones(shape)
        return tf.Variable(b)
