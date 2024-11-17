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
    """OneNearestNeighbor classifier built with Euclidean distance.

    This classifier finds the training example which is the closest
    to a new given input and assigns its label to the new input.

    Attributes :
    ------------
    X_ : numpy.ndarray
        Training data.
    y_ : numpy.ndarray
        Training labels.
    classes_ : numpy.ndarray
        Unique classes in the target data.
    n_features_in : int
        Number of features in the input data.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : numpy.ndarrayof shape (n_samples, n_features)
            The input data.
        y : numpy.ndarrayof shape (n_samples,)
            The target classes.
        Returns
        -------
        self : object
            The fitted estimator
        """
        # some checks to verify the input data
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        # storing the data
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the newly provided data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted class labels for each input instance.
        """
        # we first verify that fit has been used before
        check_is_fitted(self)
        # some checks to validate the input
        X = check_array(X)

        # initializing y_pred with a default label
        # (not mandatory but for clarity)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        # now computing prediction
        for i, x in enumerate(X):
            # first computing distances to all training instances
            distances = np.linalg.norm(self.X_ - x, axis=1)
            # Finally looking for the one with the min distance
            nearest_neighbor_index = np.argmin(distances)
            y_pred[i] = self.y_[nearest_neighbor_index]
        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input samples.
        y : numpy.ndarray of shape (n_samples,)
            The true labels of X.
        Returns
        -------
        score : float
            The accuracy score of the classifier.
        """
        # some checks to verify the inputs
        X, y = check_X_y(X, y)

        # predicting and computing accuracy
        y_pred = self.predict(X)
        y_pred = (y_pred == y).astype(int) / len(y_pred)

        return y_pred.sum()
