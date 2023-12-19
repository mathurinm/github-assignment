"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.

The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`.

We also ask to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstring similar to the one in `numpy_questions`
for the methods you code and for the class. The docstring will be checked using
`pydocstyle` that you can also call at the root of the repo.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the model using training data (X) and target values (y).

        Parameters
        ----------
        Self: refers to the instance of the class
            Used to access class attributes and methods
            Used to call other methods

        X : ndarray of shape (n_samples, n_features)
            The input design matrix.

        y: 1darray of shape(n_samples, )
            The input target value

        Returns
        -------
        Self: Returning the instance of the class on
        which method was called Enables method chaining
            Returns references to the instance
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self._X_train = X
        self._y_train = y

        return self

    def predict(self, X):
        """
        Use to predict the label of class of the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            The X_test data set used for prediction

        Returns:
        --------
        y_pred: numpy.ndarray
            The label predictions for each data points in X_test

        Notes:
        ------
        This method predicts the label of class of the input data based
        on the fitted model. The predictions are based on 1NN algorithm.

        The input X should be a 2-dimensional numpy array, where each row
        represents a single data point, and the number of columns matches the
        number of features used during model training.

        The output y_pred is a numpy array of the same length as X, containing
        the predicted class labels for each corresponding data point in X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i, x in enumerate(X):
            nearestidx = np.argmin(np.linalg.norm(self._X_train - x, axis=1))
            predictionx = self._y_train[nearestidx]
            y_pred[i] = predictionx
        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy score of the 1kk classifier on the given data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data for which to calculate the accuracy.
            the X_test dataset

        y : numpy.ndarray
            The true class labels corresponding to the input data.
            y_test array

        Returns:
        --------
        float
            The accuracy of the classifier as a value between 0 and 1

        Notes:
        ------
        This method calculates the accuracy of the classifier on the provided
        data by comparing the predicted class labels with true class labels.

        The input X should be a 2-dimensional numpy array where each row
        represents a data point, and the number of columns matches the
        number of features used during model training.

        The output y_pred is a numpy array of the same length as X, containing
        the predicted class labels for each corresponding data point in X.

        The accuracy is calculated as the proportion of correctly predicted
        class labels to the total number of data points.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        mask = y_pred == y
        y_pred = mask.astype(int) / len(X)

        return y_pred.sum()
