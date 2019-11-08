
import numpy as np
from keras.datasets import cifar10
import random
from keras.utils import np_utils


def get_cifar_data(source='download'):

    """

    Args:
        source:

    Returns:

    """

    if source == 'download':
        try:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        except:
            print("""Download error, check internet connection, \n
            or folow instructions here:
            https://stackoverflow.com/a/36815623""")

    return (x_train, y_train), (x_test, y_test)


def process_data(x_train,
                 y_train,
                 x_test,
                 y_test,
                 n_samples=None):

    """

    Args:
        x_train:
        y_train:
        x_test:
        y_test:
        n_samples:

    Returns:

    """

    num_train, height, width, depth = x_train.shape

    if n_samples is not None:
        # Use a random sample for testing
        try:
            n_samples = int(n_samples)
        except ValueError:
            print("n_samples must be positive integer")

        rand_idx_train = random.sample(range(num_train), k=n_samples)
        x_train = x_train[rand_idx_train]
        y_train = y_train[rand_idx_train]

    num_classes = np.unique(y_train.tolist()).shape[
        0]  # there are 10 image classes

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= np.max(x_train)  # Normalise data to [0, 1] range
    x_test /= np.max(x_test)  # Normalise data to [0, 1] range

    y_train = np_utils.to_categorical(y_train,
                                      num_classes)  # One-hot encode the labels
    y_test = np_utils.to_categorical(y_test,
                                     num_classes)  # One-hot encode the labels

    return x_train, y_train, x_test, y_test


def get_data_dims(x_train,
                  x_test,
                  y_train):

    """
    Args:
        x_train:
        x_test:
        y_train:

    Returns:

    """

    num_train, height, width, depth = x_train.shape
    num_test = x_test.shape[0]
    num_classes = np.unique(y_train.tolist()).shape[0]
    N = x_train.shape[0]


    return height, width, depth, num_train, num_test, num_classes, N


def accuracy(confusion_matrix):
    acc = sum(np.diag(confusion_matrix))/sum(np.array(confusion_matrix)).sum()
    return acc

