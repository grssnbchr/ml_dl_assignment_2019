import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.pipeline import Pipeline
from mpl_toolkits.axes_grid1 import make_axes_locatable

from customtransformers import NDStandardScaler, StatisticsExtractor, AddNVDI, RGB2GrayTransformer, Flattener

import warnings
from sklearn.exceptions import DataConversionWarning
# suppress certain warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning) # TODO doesnt work

def main():

    # 1. Load train data
    (X_train, y_train) = load_train_data(0.01)
    sample_size = len(X_train)
    # create a 20% hold-out-validation set
    X_train, X_val, y_train, y_val = create_validation_set(X_train, y_train, fraction=0.2,
                                                           show_class_balance=True)
    # # Preprocess
    # # Add NVDI
    # nvdiadder = AddNVDI()
    # X_train = nvdiadder.transform(X_train)
    #
    # # plot_sample_images(X_train, y_train, 4)
    # # plot_sample_channels(X_train, y_train, 6)
    #
    # # Standardize
    # standardizer = NDStandardScaler()
    # X_train = standardizer.fit_transform(X_train)
    #
    # # Try to classify with grayscale images only
    # # grayifier = RGB2GrayTransformer()
    # # X_train = grayifier.fit_transform(X_train)
    #
    #
    # # # Extract statistics
    # extract = StatisticsExtractor()
    # X_train = extract.transform(X_train)
    #
    # assert X_train.shape == (sample_size, 2, 5)
    # train_and_evaluate_model(X_train, y_train, show_class_balance=False)

    # plot_sample_images(X_train, y_train, number=8)

    # convert back from one-hot-encoding
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)
    # build a pipeline
    # TODO: add memory https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation
    pipe = Pipeline([
        ('nvdiadder', AddNVDI()),
        ('standardizer', NDStandardScaler()),
        ('statsextractor', StatisticsExtractor()),
        ('flattener', Flattener()),
        ('rf', RandomForestClassifier())
    ])
    pipe.get_params()

    param_grid = [
        {'nvdiadder': [None, AddNVDI()], # variation: add NVDI or not,
         'standardizer': [None, NDStandardScaler()], # variation: add NDStandardScaler or not
         'rf__max_features': [3],
         'rf__n_estimators': [10]}
    ]
    # TODO: apply OOB, as RF already has this https://scikit-learn.org/stable/modules/grid_search.html#out-of-bag-estimates
    grid = GridSearchCV(pipe,
                        param_grid,
                        cv=5,
                        n_jobs=-1,
                        scoring='accuracy',
                        verbose=1,
                        error_score=0,
                        return_train_score=True)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_params_)
    # predict on hold-out validation set using best estimator
    best_pred = grid.predict(X_val)
    print(classification_report(y_val, best_pred))
    plot_confusion_matrix(y_val, best_pred)

###############################################################################
###   modelling functions
###############################################################################

def train_and_evaluate_model(X_train, y_train, show_class_balance=True):
    """
    splits the training set into 80% training and 20% validation set (1-fold-cv)
    trains a simple random forest with 100 trees on the data and outputs the validation accuracy
    :param X_train: the (unflattened) X
    :param y_train: the one-hot-encoded y
    :return:
    """
    # split up into training and validation set
    X_train, X_val, y_train, y_val = create_validation_set(X_train, y_train, fraction=0.2,
                                                           show_class_balance=show_class_balance)
    # flatten everything but the first dimension
    X_train = X_train.reshape((-1, np.prod(X_train.shape[1:])))
    # convert back from one-hot-encoding
    y_train = np.argmax(y_train, axis=1)
    # Train simple classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    # evaluate
    y_pred = rf_clf.predict(X_val)
    print('Percentage correct: ', 100 * accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_pred, y_val))
    plot_confusion_matrix(y_pred, y_val)

###############################################################################
###   utility functions
###############################################################################
def create_validation_set(X_train, y_train, fraction=0.5, show_class_balance=True):
    """
    splits the training data into a training and validation set
    :param X_train: the features
    :param y_train: the one-hot-encoded labels
    :param fraction: the fraction of the test set
    :return: a tuple (X_train, X_val, y_train, y_val)
    """
    # split into train and test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=fraction,
        shuffle=True,
        random_state=42
    )
    if show_class_balance is True:
        yt = np.argmax(y_train, axis=1)
        yv = np.argmax(y_val, axis=1)
        # Plot distribution
        plt.suptitle('relative distributions of classes in train and validation set')
        plot_label_distribution_bar(yt, loc='left')
        plot_label_distribution_bar(yv, loc='right')
        plt.legend([
            'train ({0} photos)'.format(len(y_train)),
            'test ({0} photos)'.format(len(y_val))
        ])
        plt.show()
    return X_train, X_val, y_train, y_val


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


###############################################################################
###   plotting functions
###############################################################################
def plot_confusion_matrix(y_pred, y_val):
    """
    plots three confusion matrices: raw, percentage and with diagonals zeroed out
    adapted from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
    :param y_pred: predicted classes
    :param y_val: true classes
    """
    cmx = confusion_matrix(y_pred, y_val)
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()

    np.fill_diagonal(cmx_zero_diag, 0)

    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(6)) for a in ax]
    [a.set_yticks(range(6)) for a in ax]

    im1 = ax[0].imshow(cmx)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag)
    ax[2].set_title('% and 0 diagonal')

    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1)
                        for divider in dividers]

    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
    plt.show()

def plot_label_distribution_bar(y, loc='left', relative=True):
    """
    plots a grouped bar chart of the y labels
    adapted from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
    :param y: the train or validation labels as integer scalars
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
    unique, counts = np.unique(y, return_counts=True)
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
