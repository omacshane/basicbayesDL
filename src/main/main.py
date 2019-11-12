import sys
import pandas as pd
import numpy as np

sys.path.append('..')
sys.path.append('.')

from src.controller.train_model import train_model
from src.utils.data_utils import accuracy
from src.utils.metrics import calulate_average_precisions


def run_single_experiment(data_samples=1000,
                   monte_carlo_samples=10,
                   default_class=1,):
    """

    Args:
        data_samples: number of datas amples to use in training
        monte_carlo_samples: number of MC dropout iterations at test time
        default_class: (int) return AP scores for this class

    Returns:
        y_true: ground ruth class labels
        prediction_df: pandas dataframe of 'deterministic' predictions
        stochastic_predictions: pandas dataframe of mean posterior predictions

    """

    y_test, prediction_df, stochastic_predictions, Yt_hat = train_model(n_data_samples=data_samples,
                                       n_monte_carlo_samples=monte_carlo_samples)

    y_true = pd.Series([int(x)  for x in y_test])

    bayesian_predictions = stochastic_predictions.apply(lambda x: np.argmax(x),
                                                        axis=1)
    prediction = prediction_df.apply(lambda x: np.argmax(x), axis = 1)

    confuse_mat_standard = pd.crosstab(y_true, prediction)
    confuse_mat_bayesian = pd.crosstab(y_true, bayesian_predictions)

    print("Standard confusion matrix: \n {}".format(confuse_mat_standard))
    print("Bayesian confusion matrix: \n {}".format(confuse_mat_bayesian))

    print("Standard accuracy: {}".format(accuracy(confuse_mat_standard)))
    print("Bayesian accuracy: {}".format(accuracy(confuse_mat_bayesian)))


    standard_ap, stochastic_ap = calulate_average_precisions(default_class,
                                                             y_true,
                                                             prediction_df,
                                                             stochastic_predictions)

    print("Average standard precision on Class {}: {}".format(default_class,
                                                              standard_ap))
    print("Average Bayesian precision on Class {}: {}".format(default_class,
                                                              stochastic_ap))

    return y_true, prediction_df, stochastic_predictions, Yt_hat



if __name__ == "__main__":

    run_single_experiment()