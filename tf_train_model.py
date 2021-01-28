import tensorboard
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tf_network import neuralNetwork
from data_object import provide_data
import datetime
import time
import os
from cifar_10 import Cifar_10


def train_net(net, batch_size, epoch, train_db, val_db, summary_writer):
    '''
    Train your network.
    The model will be saved as .bp format in ./model dictionary.
    Input parameter:
        - net: your network
        - batch_size: the number of samples in one forward processing
        - epoch: train your network epoch time with your dataset
        - train_db: training dataset
        - val_db: validation dataset
        - summary_writer: summary object
    '''

    # create session
    sess.run(tf.global_variables_initializer())
    train_samples = train_db.num_samples  # get number of samples
    train_images = train_db.images  # get training images
    train_labels = train_db.labels  # get training labels, noting that it is one_hot format

    print("=" * 50)
    print()
    print("Start training...\n")

    # start training
    global_step = 0  # record total step when training
    for i in range(epoch):
        total_loss = 0
        for offset in range(0, train_samples, batch_size):
            # "offset" is the start position of the index, "end" is the end position of the index.
            end = offset + batch_size
            batch_train_images, batch_train_labels = train_images[
                offset:end], train_labels[
                    offset:
                    end]  # get images and labels according to the batch number
            _, loss, loss_summary = sess.run(
                [training_operation, loss_operation, merge_summary],
                feed_dict={
                    input: batch_train_images,
                    labels: batch_train_labels
                })
            total_loss += loss

            # record summary
            # print(global_step)
            summary_writer.add_summary(loss_summary, global_step=global_step)
            global_step += 1

        validation_accuracy = test_net(net, batch_size, val_db)
        loss_avg = total_loss * batch_size / train_samples
        print(
            "EPOCH {:>3d}: Loss = {:.5f}, Validation Accuracy = {:.5f}".format(
                i + 1, loss_avg, validation_accuracy))

    # save model
    if os.path.exists('./model') == False:
        os.makedirs('./model')
    output_graph_def = convert_variables_to_constants(
        sess, sess.graph_def,
        output_node_names=['output', 'loss', 'accuracy'])  # set saving node
    with tf.gfile.FastGFile('model/model.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())

    print("=" * 50)
    print()
    print("The model have been saved to ./model dictionary.")


def test_net(net, batch_size, dataset):
    '''
    Test your model with dataset.
    Input parameter:
        - net: your network
        - batch_size: the number of samples in one forward processing
        - dataset: you can choose validation or test dataset
    '''

    num_samples = dataset.num_samples
    # print(num_samples)
    data_images = dataset.images
    data_labels = dataset.labels

    total_accuracy = 0
    for offset in range(0, num_samples, batch_size):
        # "offset" is the start position of the index, "end" is the end position of the index.
        end = offset + batch_size
        batch_images, batch_labels = data_images[offset:end], data_labels[
            offset:end]  # get images and labels according to the batch number
        total_accuracy += sess.run(accuracy_operation,
                                   feed_dict={
                                       input: batch_images,
                                       labels: batch_labels
                                   })

    return total_accuracy * batch_size / num_samples


if __name__ == "__main__":

    # create tensorboard environment
    '''
    To use tensorboard,
    1. enter this code in the terminal: 
        tensorboard --logdir=./logs
    2. open url address in your browser
    '''

    # record program start time
    program_start_time = time.time()

    # create session
    with tf.Session() as sess:

        # create summary environment
        current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
        log_dir = 'logs/' + current_time

        # parameter configuration
        lr = 0.001  # learning rate
        batchsz = 256  # batch size
        epoch = 15  # training period

        # prepare training dataset and test dataset
        # train: 55000, test: 10000, validation: 5000
        cifar_10 = Cifar_10()
        f_path = "data/cifar-10-batches-py/"
        cifar_10.read_data_sets(f_path)  # load cifar-10 dataset

        # mnist = input_data.read_data_sets('mnist_data/')  # load minist dataset
        data = provide_data(cifar_10)

        # create input and output placeholder
        input = tf.placeholder(dtype=tf.float32,
                               shape=[None, 32, 32, 1],  # revised by me
                               name='input')
        labels = tf.placeholder(dtype=tf.float32,
                                shape=[None, 10],
                                name='labels')

        # create instance of neural network
        net = neuralNetwork()

        # forward the network
        logits = net.forward(input)

        # get loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
        loss_operation = tf.reduce_mean(cross_entropy, name="loss")

        # set up the optimizer and optimize the parameters
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        training_operation = optimizer.minimize(loss_operation)

        # post-processing, get accuracy
        prediction = tf.argmax(logits, axis=1, name='output')
        correct_prediction = tf.equal(prediction, tf.argmax(labels, axis=1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,
                                                    tf.float32),
                                            name="accuracy")

        # create summary scalar
        tf.summary.scalar('Loss', loss_operation)
        tf.summary.scalar('Accuracy', accuracy_operation)
        merge_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # record start training time
        start_training_time = time.time()
        # start training
        train_net(net, batchsz, epoch, data.train, data.validation,
                  summary_writer)
        print("Training time: {:.3f}s.\n".format(time.time() -
                                                 start_training_time))

        # record start testimg time
        start_testing_time = time.time()
        # test model accuracy
        print("=" * 50)
        print("\nStart testing...")
        acc = test_net(net, batchsz, data.test)
        print("Test Accuracy = {:.5f}".format(acc))
        print("Testing time: {:.5f}s\n".format(time.time() -
                                               start_testing_time))

    # output program end time
    print("Program running time: {:.3f}s.".format(time.time() -
                                                  program_start_time))
