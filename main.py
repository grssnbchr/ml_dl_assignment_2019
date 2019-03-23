import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from sklearn.ensemble import RandomForestClassifier


def main():
    # 1. Load train data
    (X_train, y_train) = load_train_data(0.01)


    #
    # X_test = load_data("test_features_spam.csv")
    # y_test = load_data("test_labels_spam.csv").ravel()
    #
    # # 2. Train different models
    # algorithms = [
    #     SVC(**params_svm_linear),
    #     SVC(**params_svm_rbf),
    #     LogisticRegression(),
    #     MultinomialNB(**params_naive_bayes),
    #     RandomForestClassifier(**params_random_forest),
    #     AdaBoostClassifier(**params_ada_boost)]
    # classifiers = train_models(algorithms, X_train, y_train)
    #
    # # 3. Apply the trained models
    # apply_models(classifiers, X_test, y_test)
    #
    # # 3.1 Apply majority vote ensemble of ensembles
    # predictions = apply_models_majority_vote(classifiers, X_test)
    # test_accuracy = calc_accuracy(predictions, y_test)
    # print('Testing with majority vote (aggregated):')
    # print('\tAccuracy with %d %ss: %.5f' % (len(classifiers), classifiers[0].__class__.__name__, test_accuracy))
    #
    # predictions = apply_models_majority_vote(classifiers, X_test, aggregated=False)
    # test_accuracy = calc_accuracy(predictions, y_test)
    # print('Testing with majority vote (not aggregated):')
    # print('\tAccuracy with %d %ss: %.5f' % (len(classifiers), classifiers[0].__class__.__name__, test_accuracy))
    #
    # # 4. Write predictions to submission file
    # write_predictions('submission_spam.csv', X_test, predictions)
    #
    # # 5.-7. [Optional] Further investigations on own emails in text file format
    # # apply_to_own_mails(classifiers)


###############################################################################
###   utility function to load data
###############################################################################
def load_train_data(fraction=1):
    '''
    :param dataset_name: defines the csv file to be loaded
    :param fraction: load only a fraction of the data
    :return: the pre-processed X and y data as a tuple of tow numpy array of
    shape (sample_size, x, y, channels) and (sample_size, 6), respectively
    '''
    # X
    row_number = 324000 # TODO: adapt to full size
    rows = math.ceil(row_number * fraction)
    X_df = pd.read_csv('data/deepsat-sat6/X_train_sat6.csv', header=None, sep=',', nrows=rows)
    # unfold
    X_np = np.array(X_df)
    X_train = X_np.reshape((-1,
                            28,
                            28,
                            4))
    y_df = pd.read_csv('data/deepsat-sat6/y_train_sat6.csv', header=None, sep=',', nrows=rows)
    y_train = np.array(y_df)
    return (X_train, y_train)

if __name__ == '__main__':
    main()