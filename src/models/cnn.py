import sys
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.regularizers import l2

import keras.backend as K

sys.path.append('..')
sys.path.append('.')

from src.config import model_parameters as pars
import src.utils.data_utils as data_utils


batch_size = pars['batch_size']
num_epochs =pars['num_epochs']
kernel_size = pars['kernel_size']
pool_size = pars['pool_size']
conv_depth_1 = pars['conv_depth_1']
conv_depth_2 = pars['conv_depth_2']
drop_prob_1 = pars['drop_prob_1']
drop_prob_2 = pars['drop_prob_2']
hidden_size = pars['hidden_size']
tau = pars['tau']
lengthscale = pars['lengthscale']
dropout = pars['dropout']


def create_model(height,
                 width,
                 depth,
                 num_classes,
                 N):
    """

    Args:
        height: number pixels along y axis
        width: number pixels along x axis
        depth: number of channels
        num_classes: number of target classes
        N: number of training observations

    Returns:
        model: a Keras model

    """

    reg = lengthscale ** 2 * (1 - dropout) / (2. * N * tau)  # from paper

    ### Model
    # @ convolutional layers  and two dense layers
    inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                           padding='same',
                           activation='relu')(inp)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                           padding='same',
                           activation='relu')(drop_1)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    flat = Flatten()(drop_2)
    bn = BatchNormalization()(flat)
    hidden = Dense(hidden_size, activation='relu',
                   kernel_regularizer= l2(reg))(bn)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(inputs=inp, outputs=out)

    return model