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
    """One Nearest Neighbor classifier.

    This classifier predicts the label of the closest training sample
    using the Euclidean distance.
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : object
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
        """Predict the closest label for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["X_", "y_", "n_features_in_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has {} features, but OneNearestNeighbor was fitted with "
                "{} features.".format(X.shape[1], self.n_features_in_)
            )

        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=self.y_.dtype)

        for i in range(n_samples):
            # Compute Euclidean distances to all training samples
            distances = np.linalg.norm(self.X_ - X[i], axis=1)
            nn_index = np.argmin(distances)
            y_pred[i] = self.y_[nn_index]

        return y_pred

    def score(self, X, y):
        """Return accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy (fraction of correct predictions).
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
