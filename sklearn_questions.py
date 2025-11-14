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
for the methods you code and for the class. The docstring will be checked
using `pydocstyle` that you can also call at the root of the repo.
"""
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest-neighbor classifier.

    This classifier predicts, for each input sample, the target of the
    closest training sample in Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        """Initialize the OneNearestNeighbor classifier."""
        # This estimator has no hyper-parameters.
        pass

    def fit(self, X, y):
        """Fit the one-nearest-neighbor classifier.

        The training samples and their labels are stored so that predictions
        can be made by finding the closest training sample to each new point.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        # validate_data will set n_features_in_ and perform basic checks
        X, y = validate_data(self, X, y)
        check_classification_targets(y)

        # Store training data and targets
        self.X_ = X
        self.y_ = y

        # Attributes expected by scikit-learn
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Each sample in X is assigned the label of the closest training sample
        stored during :meth:`fit`, using the Euclidean distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample in X.
        """
        check_is_fitted(self, attributes=["X_", "y_"])
        # reset=False enforces consistency with n_features_in_
        X = validate_data(self, X, reset=False)

        # Compute squared Euclidean distances to all training samples:
        # diff shape: (n_samples_test, n_samples_train, n_features)
        diff = X[:, np.newaxis, :] - self.X_[np.newaxis, :, :]
        distances = np.sum(diff**2, axis=2)

        # Index of nearest neighbor in the training set for each test sample
        nearest_idx = np.argmin(distances, axis=1)
        y_pred = self.y_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions on X with respect to y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))