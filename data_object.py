import numpy as np
import collections
from sklearn.utils import shuffle
import cv2
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
    train_images = np.zeros([train_images_org.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
    for i in range(train_images_org.shape[0]):
        train_images[i] = cv2.resize(train_images_org[i].astype('uint8'), dsize=(IMAGE_SIZE, IMAGE_SIZE))
    train_labels = cifar_10.train.labels

    validation_images_org = cifar_10.validation.images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    validation_images = np.zeros([validation_images_org.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
    for i in range(validation_images_org.shape[0]):
        validation_images[i] = cv2.resize(validation_images_org[i].astype('uint8'), dsize=(IMAGE_SIZE, IMAGE_SIZE))
    validation_labels = cifar_10.validation.labels

    test_images_org = cifar_10.test.images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_images = np.zeros([test_images_org.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
    for i in range(test_images_org.shape[0]):
        test_images[i] = cv2.resize(test_images_org[i].astype('uint8'), dsize=(IMAGE_SIZE, IMAGE_SIZE))
    test_labels = cifar_10.test.labels

    # i = 0
    # while (i < 1000):
    #     plt.imshow(test_images[i])
    #     plt.show()
    #     i += 1

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


if __name__ == "__main__":
    file = "data/cifar-10-batches-py/data_batch_" + str(1)
    part_1 = unpickle(file)
    # data.labels
    # data.data (10000, 3072)  3072 = 32*32*3
    # cifar.image = part_1[data]
    pass
    # cifar_10 = part1
