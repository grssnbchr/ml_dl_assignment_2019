import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from customtransformers import NDStandardScaler, StatisticsExtractor, AddNVDI


def main():
    # 1. Load train data
    (X_train, y_train) = load_train_data(0.01)
    sample_size = len(X_train)

    baseline_model(X_train, y_train)

    # # 2. Preprocess
    # # Add NVDI
    # nvdiadder = AddNVDI()
    # X_train = nvdiadder.transform(X_train)
    #
    # plot_sample_images(X_train, y_train, 4)
    # plot_sample_channels(X_train, y_train, 6)
    #
    # # Standardize
    # standardizer = NDStandardScaler()
    # X_train = standardizer.transform(X_train)
    # # Extract statistics
    # extract = StatisticsExtractor()
    # X_train = extract.transform(X_train)
    #
    # assert X_train.shape == (sample_size, 2, 5)

    # Extract features

    # plot_sample_images(X_train, y_train, number=8)

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
###   modelling functions
###############################################################################

def baseline_model(X_train, y_train):
    """
    splits the training set into 80% training and 20% validation set (1-fold-cv)
    trains a simple random forest with 100 trees on the data and outputs the validation accuracy
    :param X_train: the (unflattened) X
    :param y_train: the one-hot-encoded y
    :return:
    """
    # flatten everything but the first dimension
    X_train = np.transpose(X_train.reshape(-1, X_train.shape[0]))
    # split into train and test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )
    # Plot distribution
    plt.suptitle('relative distributions of classes in train and validation set')
    plot_label_distribution_bar(y_train, loc='left')
    plot_label_distribution_bar(y_val, loc='right')
    plt.legend([
        'train ({0} photos)'.format(len(y_train)),
        'test ({0} photos)'.format(len(y_val))
    ])
    plt.show()
    # convert back from one-hot-encoding
    y_train = np.argmax(y_train, axis=1)
    # Train simple classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    # evaluate
    y_pred = rf_clf.predict(X_val)
    print('Percentage correct: ', 100 * np.sum(y_pred == y_val) / len(y_val))


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
    returns the colloquial label associated with class y
    :param y: the one-hot encoded label
    :return: the colloquial name of the label
    """
    annotations = pd.read_csv('data/deepsat-sat6/sat6annotations.csv', header=None)

    return annotations[annotations[np.argmax(y) + 1] == 1][0].item()


def plot_label_distribution_bar(y, loc='left', relative=True):
    """
    plots a grouped bar chart of the y labels
    adapted from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
    :param y: the train or validation labels as one-hot encoded vectors
    :param loc: left or right bar
    :param relative: if True percentages are shown
    :return:
    """
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5

    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]

    if relative:
        # plot as a percentage
        counts = 100 * counts[sorted_index] / len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'

    xtemp = np.arange(len(unique))

    plt.bar(xtemp + n * width, counts, align='center', alpha=.7, width=width)
    # one-hot encode labels and extract colloquial names
    enc = OneHotEncoder(sparse=False, categories='auto')
    labels = list(
        np.apply_along_axis(
            arr=enc.fit_transform(unique.reshape(len(unique), 1)),
            func1d=get_label, # extract colloquial names
            axis=0
        )
    )
    plt.xticks(xtemp, labels)
    plt.xlabel('land cover class')
    plt.ylabel(ylabel_text)


def plot_sample_images(tX, tY, number=4):
    """
    plots a number of sample images with the according label
    :param tX: training data
    :param tY: one-hot encoded labels
    :param number: the number of images to plot
    """
    assert number >= 4
    fig, m_axs = plt.subplots(4, number // 4, figsize=(4, 4))
    for (x, y, c_ax) in zip(tX, tY, m_axs.flatten()):
        print(x.shape)
        print(x[1:3, 1:3, :3])
        c_ax.imshow(x[:, :, :3].astype(np.uint8),  # since we don't want NIR in the display
                    interpolation='none')
        c_ax.axis('off')
        c_ax.set_title('Cat:{}'.format(get_label(y)))
    plt.show()


def plot_sample_channels(tX, tY, number=3):
    """
    :param tX: input data
    :param tY: labels
    :param number: number of different examples
    """
    assert tX.shape[3] == 5  # assert 5 channels
    fig, m_axs = plt.subplots(number, 6)
    imgs_to_plot = zip(tX, tY, m_axs[:, 0])
    for i, (x, y, c_ax) in enumerate(imgs_to_plot):
        c_ax.imshow(x[:, :, :3].astype(np.uint8),  # since we don't want NIR and NVDI in the composite image
                    interpolation='none')
        c_ax.axis('off')
        c_ax.set_title('Cat:{}'.format(get_label(y)))
        for (j, r_ax) in zip(range(0, 5), m_axs[i, 1:]):
            r_ax.imshow(x[:, :, j],
                        interpolation='none',
                        cmap='gray')
            r_ax.axis('off')
            channels = ['red', 'green', 'blue', 'nir', 'nvdi']
            r_ax.set_title(f'Ch:{channels[j]}')

    plt.show()


if __name__ == '__main__':
    main()
