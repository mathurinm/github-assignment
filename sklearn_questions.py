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


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        This stores the training data so that predictions can be made
        by looking for the closest training sample to each new point.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target labels for each training sample.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict class labels for the given samples.

        For each sample in X, the predicted label is the label of the
        closest training sample (in Euclidean distance).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=self.y_.dtype)

        for i in range(n_samples):
            x = X[i]
            dists = np.linalg.norm(self.X_ - x, axis=1)
            nn_idx = np.argmin(dists)
            y_pred[i] = self.y_[nn_idx]

        return y_pred

    def score(self, X, y):
        """Compute accuracy of the classifier on the given test data.

        The accuracy is the proportion of correctly classified samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test input samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions on X compared to y.
        """
        X, y = check_X_y(X, y)
        check_is_fitted(self)

        y_pred = self.predict(X)
        return np.mean(y_pred == y)
