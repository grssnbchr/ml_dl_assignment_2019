import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


def main():
    # 1. Load train data
    (X_train, y_train) = load_train_data(0.01)

    # plot_sample_images(X_train, y_train, number=8)

    plot_sample_channels(X_train, y_train, 6)
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
###   utility functions
###############################################################################
def load_train_data(fraction=1):
    """
    :param fraction: load only a fraction of the data
    :return: the pre-processed X and y data as a tuple of tow numpy array of
    shape (sample_size, x, y, channels) and (sample_size, 6), respectively
    """
    # X
    row_number = 324000
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


def get_label(y):
    """
    :param y: the one-hot encoded label
    :return: the colloquial name of the label
    """
    annotations = pd.read_csv('data/deepsat-sat6/sat6annotations.csv', header=None)
    return annotations[annotations[np.argmax(y) + 1] == 1][0].item()


def plot_sample_images(tX, tY, number=4):
    """
    plots a number of sample images with the according label
    :param tX: training data
    :param tY: one-hot encoded labels
    :param number: the number of images to plot
    """
    fig, m_axs = plt.subplots(4, number // 4, figsize=(4, 4))
    for (x, y, c_ax) in zip(tX, tY, m_axs.flatten()):
        c_ax.imshow(x[:, :, :3],  # since we don't want NIR in the display
                    interpolation='none')
        c_ax.axis('off')
        c_ax.set_title('Cat:{}'.format(get_label(y)))
    plt.show()


def plot_sample_channels(tX, tY, number=3):
    """
    :param tX:
    :param tY:
    :param number:
    """
    fig, m_axs = plt.subplots(number, 5)
    imgs_to_plot = zip(tX, tY, m_axs[:, 0])
    for i, (x, y, c_ax) in enumerate(imgs_to_plot):
        c_ax.imshow(x[:, :, :3],  # since we don't want NIR in the display
                    interpolation='none')
        c_ax.axis('off')
        c_ax.set_title('Cat:{}'.format(get_label(y)))
        for (j, r_ax) in zip(range(0, 4), m_axs[i, 1:]):
            r_ax.imshow(x[:, :, j],
                        interpolation='none',
                        cmap='gray')
            r_ax.axis('off')
            channels = ['red', 'green', 'blue', 'nir']
            r_ax.set_title(f'Ch:{channels[j]}')

    plt.show()

if __name__ == '__main__':
    main()
