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
        """Fits the OneNearestNeighbour function to the training data (X, y).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input data.

        y : ndarray of size (n_samples,)
            The target values as integers or strings.

        Returns
        ----------
        self : OneNearestNeighbor classifier
            The OneNearestNeighbor classifier fitted on the training data
            (X, y).

        Raises
        ----------
        ValueError
            If the input X is not a numpy array or
            if the shape is not 2D.
            If the input y is not a numpy array or
            if the shape is not 1D.
            If the number of samples in X and y are not equal.
            If X or y contain non-finite values.
        ValueError
            If the input y is consistent with a classification task. Checks tha
            t y does not contain continuous values, but rather class labels.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predicts the label of the new observations
        with features given by the input array X.

        Uses the fitted OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_new_samples, n_features)
            The features of the new observations to be predicted.

        Returns
        ----------
        y_pred : ndarray of shape (n_new_samples,)
            The predicted labels of the new observations whose features are
            given by the input array X.

        Raises
        ----------
        NotFittedError
            If the OneNearestNeighbor classifier is not fitted by checking if
            the attributes X_ and y_ are defined.
        ValueError
            If the input X is not a numpy array or
            if the shape is not 2D or
            if X containes non-finite values.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_ - x, axis=1)
            nearest_neighbor_index = np.argmin(distances)
            y_pred[i] = self.y_[nearest_neighbor_index]

        return y_pred

    def score(self, X, y):
        """Write docstring.

        And describe parameters
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        score = 0

        for i, prediction in enumerate(y_pred):
            if prediction == y[i]:
                score += 1

        return score/len(y)
