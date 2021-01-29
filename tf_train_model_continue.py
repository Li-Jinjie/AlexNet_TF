import tensorboard
import tensorflow as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.python.tools import freeze_graph
from tf_network import neuralNetwork
from data_object import provide_data
from data_object import preprocess
import datetime
from tqdm import tqdm
import time
import os
from cifar_10 import Cifar_10


def train_net(net, batch_size, epochs, train_db, val_db, summary_writer):
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
    # sess.run(tf.compat.v1.global_variables_initializer())

    train_samples = train_db.num_samples  # get number of samples
    train_images = train_db.images  # get training images
    train_labels = train_db.labels  # get training labels, noting that it is one_hot format

    print("=" * 50)
    print()
    print("Start training...\n")

    # start training
    global_step = 0  # record total step when training
    for epoch in range(epochs):
        total_loss = 0
        for offset in tqdm(range(0, train_samples, batch_size)):
            # "offset" is the start position of the index, "end" is the end position of the index.
            end = offset + batch_size
            batch_train_images, batch_train_labels = train_images[offset:end], train_labels[offset:end]
            # get images and labels according to the batch number

            batch_train_images = preprocess(batch_train_images, IMAGE_SIZE)

            _, loss, loss_summary = sess.run(
                [training_operation, loss_operation, merge_summary],
                feed_dict={
                    input: batch_train_images,
                    labels: batch_train_labels,
                    prob: 0.5  # the probability to discard elements
                })
            total_loss += loss

            # record summary
            # print(global_step)
            summary_writer.add_summary(loss_summary, global_step=global_step)
            global_step += 1

        validation_accuracy = test_net(net, batch_size, val_db)
        loss_avg = total_loss * batch_size / train_samples
        print(
            "EPOCH {:>3d}: Loss = {:.5f}, Validation Accuracy = {:.5f}".format(epoch + 1, loss_avg,
                                                                               validation_accuracy))

        # save model for every 10 epochs
        if ((epoch + 1) != 0) and ((epoch + 1) % 10 == 0):
            ckpt_path = os.path.join(dictionary, 'ckpt_model_valid_acc=%.4f.ckpt' % validation_accuracy)
            save_path = saver.save(sess, ckpt_path, global_step=epoch, write_meta_graph=False)
            print("model has saved,saved in path: %s" % save_path)

    #### save model ####
    pbtxt_name = 'frozen_model.pbtxt'
    pbtxt_path = os.path.join(dictionary, pbtxt_name)
    frozen_model_path = os.path.join(dictionary, 'frozen_model.pb')
    output_node = 'full_layer_03/linear'

    ckpt_path = os.path.join(dictionary, 'ckpt_model_valid_acc=%.4f.ckpt' % validation_accuracy)
    save_path = saver.save(sess, ckpt_path, global_step=epoch, write_meta_graph=False)

    # This will only save the graph but the variables will not be saved.
    # You have to freeze your model first.
    tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=dictionary, name=pbtxt_name, as_text=True)
    # Freeze graph
    freeze_graph.freeze_graph(input_graph=pbtxt_path, input_saver='',
                              input_binary=False, input_checkpoint=ckpt_path, output_node_names=output_node,
                              restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                              output_graph=frozen_model_path, clear_devices=True, initializer_nodes='')

    # save the final model
    # if os.path.exists('./model') == False:
    #     os.makedirs('./model')
    # output_graph_def = convert_variables_to_constants(
    #     sess, sess.graph_def,
    #     output_node_names=['output', 'loss', 'accuracy'])  # set saving node
    # with tf.gfile.GFile('model/AlexNet_model.pb', mode='wb') as f:
    #     f.write(output_graph_def.SerializeToString())

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
    for offset in tqdm(range(0, num_samples, batch_size)):
        # "offset" is the start position of the index, "end" is the end position of the index.
        end = offset + batch_size
        batch_images, batch_labels = data_images[offset:end], data_labels[
                                                              offset:end]  # get images and labels according to the batch number

        batch_images = preprocess(batch_images, IMAGE_SIZE)

        total_accuracy += sess.run(accuracy_operation,
                                   feed_dict={
                                       input: batch_images,
                                       labels: batch_labels,
                                       prob: 0.0  # the probability to discard elements
                                   })

    return total_accuracy * batch_size / num_samples


# def load_pb(path_to_pb):
#     with tf.io.gfile.GFile(path_to_pb, "rb") as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph


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
    with tf.compat.v1.Session() as sess:
        # create summary environment
        current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
        log_dir = 'logs/' + current_time

        # parameter configuration
        # TODO: change learning rate to decayed learning rate
        lr = 0.001  # learning rate
        batchsz = 128  # batch size
        epoch = 30  # training period
        IMAGE_SIZE = 224

        # prepare training dataset and test dataset
        # train: 55000, test: 10000, validation: 5000
        cifar_10 = Cifar_10()
        f_path = "data/cifar-10-batches-py/"
        cifar_10.read_data_sets(f_path)  # load cifar-10 dataset

        # load cifar-10 dataset
        data = provide_data(cifar_10)

        # !!!  Restore the model  !!!
        # step 1
        saver = tf.train.import_meta_graph('./ckpt_model/ckpt_model_valid_acc=0.2969.ckpt.meta')
        saver.restore(sess, './ckpt_model/ckpt_model_valid_acc=0.2969.ckpt')
        # step 2
        graph = tf.get_default_graph()
        # input
        input = graph.get_tensor_by_name('input:0')
        labels = graph.get_tensor_by_name('labels:0')
        prob = graph.get_tensor_by_name('prob:0')

        dictionary = './ckpt_model'
        if os.path.exists(dictionary) == False:
            os.makedirs(dictionary)

        # create instance of neural network
        net = neuralNetwork()

        # forward the network
        logits = net.forward(input, prob)

        # get loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
        loss_operation = graph.get_tensor_by_name('loss_op:0')

        # set up the optimizer and optimize the parameters
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        training_operation = optimizer.minimize(loss_operation)

        # post-processing, get accuracy
        prediction = graph.get_tensor_by_name('output:0')
        correct_prediction = tf.equal(prediction, tf.argmax(labels, axis=1))
        accuracy_operation = graph.get_tensor_by_name('accuracy:0')

        # create summary scalar
        tf.compat.v1.summary.scalar('Loss', loss_operation)
        tf.compat.v1.summary.scalar('Accuracy', accuracy_operation)
        merge_summary = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)

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
