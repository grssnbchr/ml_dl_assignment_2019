import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class NDStandardScaler(TransformerMixin):
    """
    Applies StandardScaler, but for arrays with more than two dimensions.
    "It simply flattens the features of the input before giving it to sklearn's StandardScaler.
     Then, it reshapes them back."
    Taken from https://stackoverflow.com/a/53231071/1477035
    """
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


class AddNVDI(TransformerMixin):
    """
    Adds a fifth channel NVDI that is composed
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **kwargs):
        """returns itself"""
        return self

    def transform(self, X, y=None, **kwargs):
        """
        :param X: the numpy array to extract the NVDI from
        :return: a numpy array of shape (sample_size, rows, cols, existing_channels + 1)
        """
        X = np.array(X)
        nvdi = self._extract_nvdi(X)
        X = np.concatenate([X, nvdi], axis=-1)
        return X

    def _extract_nvdi(self, X):
        """
        :param X: the sample to be transformed
        :return: the nvdi channel as numpy array of shape (sample_size, rows, cols, 1)
        """
        return np.expand_dims((X[:, :, :, 3] - X[:, :, :, 0]) / (X[:, :, :, 3] + X[:, :, :, 0]),
                              axis=3) # (NIR - red) / (NIR + red)


class StatisticsExtractor(TransformerMixin):
    """
    Extracts mean and standard deviation for every channel in an image
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **kwargs):
        """returns itself"""
        return self


    def transform(self, X, y=None, **kwargs):
        """
        :param X: the sample to be transformed
        :param y:
        :param kwargs:
        :return: an array of shape (sample_size, 2 (mean and stdev), channels)
        """
        X = np.array(X)
        means = np.mean(X, axis=(1, 2))
        stdevs = np.std(X, axis=(1, 2))
        return np.stack([means, stdevs], axis=1)

