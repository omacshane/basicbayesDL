import sys
import pandas as pd
import numpy as np
import time

import keras.backend as K
from keras.callbacks import EarlyStopping

sys.path.append('..')
sys.path.append('.')

from src.config import model_parameters as pars
from src.utils.data_utils import get_cifar_data, get_data_dims, process_data
import src.models.cnn as cnn_model

batch_size = pars['batch_size']
num_epochs = pars['num_epochs']
T = pars['num_mc_samples']
N_SAMPLES = pars['default_n_data_samples']


def train_model(n_data_samples=N_SAMPLES,
                n_monte_carlo_samples=T):

    """

    Args:
        n_data_samples: number of datas amples to use in training
        n_monte_carlo_samples: number of MC dropout iterations at test time

    Returns:
        y_test:
        predictions_df:
        bayesian_predictions

    """

    (x_train, y_train), (x_test, y_test) = get_cifar_data()

    height, width, depth, num_train, num_test, num_classes, N = get_data_dims(x_train,
                                                                              x_test,
                                                                              y_train)

    X_train, Y_train, X_test, Y_test = process_data(x_train,
                                                    y_train,
                                                    x_test,
                                                    y_test,
                                                    n_samples=n_data_samples)



    model = cnn_model.create_model(height,
                                   width,
                                   depth,
                                   num_classes,
                                   N)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=2,
                       verbose=1, mode='auto',
                       restore_best_weights=True)

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_split=0.1,
              callbacks=[es])

    print("Evaluating Model...")
    model.evaluate(X_test, Y_test, verbose=1)

    print("Generating standard predictions...")
    standard_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

    # Imolement dropout uncertainty
    predict_stochastic = K.function([model.layers[0].input,
                                     K.learning_phase()],
                                    [model.layers[-1].output])

    print("""Generating stochastic predictions,\n
            this is the slow bit... \n""")
    start_time = time.time()
    Yt_hat = np.array([predict_stochastic([X_test, 1]) for _ in range(n_monte_carlo_samples)])
    end_time = time.time()
    print("Took this amount of minutes:")
    print((end_time - start_time)/60)

    print("Saving predictions")
    np.save("ten_sims", Yt_hat)

    prediction_df = pd.DataFrame(standard_pred)

    stoch_preds = pd.DataFrame()

    print("Calculating means etc...")
    for i in range(len(X_test)):
        stoch_preds.at[i, 0] = Yt_hat[:, 0, i, 0].mean()
        stoch_preds.at[i, 1] = Yt_hat[:, 0, i, 1].mean()
        stoch_preds.at[i, 2] = Yt_hat[:, 0, i, 2].mean()
        stoch_preds.at[i, 3] = Yt_hat[:, 0, i, 3].mean()
        stoch_preds.at[i, 4] = Yt_hat[:, 0, i, 4].mean()
        stoch_preds.at[i, 5] = Yt_hat[:, 0, i, 5].mean()
        stoch_preds.at[i, 6] = Yt_hat[:, 0, i, 6].mean()
        stoch_preds.at[i, 7] = Yt_hat[:, 0, i, 7].mean()
        stoch_preds.at[i, 8] = Yt_hat[:, 0, i, 8].mean()
        stoch_preds.at[i, 9] = Yt_hat[:, 0, i, 9].mean()

    return y_test, prediction_df, stoch_preds, Yt_hat

