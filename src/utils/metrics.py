
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def calulate_average_precisions(target_class,
                                labels,
                                predictions,
                                stochastic_predictions):

    indicator_list = [1 if x == target_class else 0 for x in labels]

    print("Calculating precision/recall")
    precision, recall, _ = precision_recall_curve(indicator_list,
                                                    predictions[target_class],
                                                    pos_label=1)
    precision_stoch, recall_stoch, _ = precision_recall_curve(indicator_list,
                               stochastic_predictions[target_class],
                               pos_label=1)


    average_precision = average_precision_score(indicator_list,
                                                predictions[target_class])

    average_precision_stoch = average_precision_score(indicator_list,
                                                      stochastic_predictions[target_class])

    return average_precision, average_precision_stoch