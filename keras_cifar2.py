import numpy as np
import pandas as pd
import keras

from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.regularizers import l2

import keras.backend as K
import random
import time

import sklearn

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

K.clear_session()

batch_size = 64 # in each iteration, we consider 32 training examples at once
num_epochs = 1 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 128 # the FC layer will have 512 neurons



X_train = np.load("/Users/Shane/PycharmProjects/BDL/cifar10/all_data/X_train.npy")
X_test = np.load("/Users/Shane/PycharmProjects/BDL/cifar10/all_data/X_test.npy")
y_train = np.load("/Users/Shane/PycharmProjects/BDL/cifar10/all_data/y_train.npy")
y_test = np.load("/Users/Shane/PycharmProjects/BDL/cifar10/all_data/y_test.npy")

num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10

rand_inc_train = random.sample(range(num_train), k = 10000)
#rand_inc_test = random.choices(range(num_test), k = 5000)

X_train = X_train[rand_inc_train]
y_train = y_train[rand_inc_train]
X_test = X_test#[rand_inc_test]
y_test = y_test#[rand_inc_test]

num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train.tolist()).shape[0] # there are 10 image classes


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels



N = X_train.shape[0]
dropout = 0.5
batch_size = 64
tau = 0.5  # obtained from BO
lengthscale = 1e-2
reg = lengthscale ** 2 * (1 - dropout) / (2. * N * tau)

#s the dataset is balanced across the ten classes); - We hold out 10% of the data for validation purposes.

inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
#conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
#conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
bn = BatchNormalization()(flat)
hidden = Dense(hidden_size, activation='relu', kernel_regularizer= l2(reg))(bn)

drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)


model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

print("Evaluating Model...")
model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

print("Generating standard predictions...")
standard_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

# Inspired by Yarin Gal
T = 25
predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

print("Generating stochastic predictions...")
start_time = time.time()
Yt_hat = np.array([predict_stochastic([X_test, 1]) for _ in range(T)])
end_time = time.time()
print("Took this amount of seconds:")
print(end_time - start_time)

print("Saving predictions")
np.save("ten_sims", Yt_hat)

preds_df = pd.DataFrame(standard_pred)

prediction = preds_df.apply(lambda x: np.argmax(x), axis = 1)

stoch_preds = pd.DataFrame()

print("Calculating means etc...")
for i in range(len(X_test)):
    stoch_preds.set_value(index = i, col= 0, value = Yt_hat[:, 0, i, 0].mean())
    stoch_preds.set_value(index=i, col=1, value=Yt_hat[:, 0, i, 1].mean())
    stoch_preds.set_value(index=i, col=2, value=Yt_hat[:, 0, i, 2].mean())
    stoch_preds.set_value(index=i, col=3, value=Yt_hat[:, 0, i, 3].mean())
    stoch_preds.set_value(index=i, col=4, value=Yt_hat[:, 0, i, 4].mean())
    stoch_preds.set_value(index=i, col=5, value=Yt_hat[:, 0, i, 5].mean())
    stoch_preds.set_value(index=i, col=6, value=Yt_hat[:, 0, i, 6].mean())
    stoch_preds.set_value(index=i, col=7, value=Yt_hat[:, 0, i, 7].mean())
    stoch_preds.set_value(index=i, col=8, value=Yt_hat[:, 0, i, 8].mean())
    stoch_preds.set_value(index=i, col=9, value=Yt_hat[:, 0, i, 9].mean())


bayesian_predictions = stoch_preds.apply(lambda x: np.argmax(x), axis = 1)

y_true = pd.Series([int(x)  for x in y_test])

confuse_mat_standard = pd.crosstab(y_true, prediction)
confuse_mat_bayesian = pd.crosstab(y_true, bayesian_predictions)

print("Standard confusion matrix: \n {}".format(confuse_mat_standard))
print("Bayesian confusion matrix: \n {}".format(confuse_mat_bayesian))

def accuracy(confusion_matrix):
    acc = sum(np.diag(confusion_matrix))/sum(np.array(confusion_matrix)).sum()
    return acc

print("Standard accuracy: {}".format(accuracy(confuse_mat_standard)))
print("Bayesian accuracy: {}".format(accuracy(confuse_mat_bayesian)))

one_true = [1 if x == 1 else 0 for x in y_test]

print("Calculating precision/recall")
precision1, recall1, _ = precision_recall_curve(one_true, preds_df[1],  pos_label=1)
precision1_stoch, recall1_stoch, _ = precision_recall_curve(one_true, preds_df['mean_pred_1'],  pos_label=1)


average_precision1 = average_precision_score(one_true, preds_df[1])
print("Average standard precision on Class 1: {}".format(average_precision1))
average_precision1_stoch = average_precision_score(one_true, preds_df['mean_pred_1'])
print("Average Bayesian precision on Class 1: {}".format(average_precision1_stoch))