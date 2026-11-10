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
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier.

    Predicts for each sample the label of the closest training sample,
    using the Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the classifier by storing the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training samples.
        y : ndarray of shape (n_samples,)
            The training labels.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # 1-NN has no real training step, we just keep the data
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the label of each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The label of the closest training sample for each sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # distances between each sample and each training sample,
        # shape (n_samples, n_train_samples)
        dists = np.linalg.norm(X[:, np.newaxis] - self.X_train_, axis=2)
        y_pred = self.y_train_[np.argmin(dists, axis=1)]
        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The test samples.
        y : ndarray of shape (n_samples,)
            The true labels.

        Returns
        -------
        score : float
            The fraction of correctly classified samples.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
