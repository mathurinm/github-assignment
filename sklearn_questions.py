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
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One nearest neighbor classifier.

    This estimator predicts the label of each sample using the label of
    the closest training sample according to the Euclidean distance.
    """

    def __init__(self):  # noqa: D107
        # No hyper-parameters for this simple estimator.
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        # validate_data sets n_features_in_ and does input validation
        X, y = validate_data(self, X, y)
        check_classification_targets(y)

        # Store training data
        self.X_train_ = X
        self.y_train_ = y

        # Attributes expected by scikit-learn
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict class labels for the provided samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)

        # Use validate_data with reset=False so n_features_in_ is checked
        X = validate_data(self, X, reset=False)

        # Compute Euclidean distance from each test point to each train point
        # X shape: (n_test, n_features)
        # self.X_train_ shape: (n_train, n_features)
        diff = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)  # (n_test, n_train)

        # Index of the nearest neighbor for each test sample
        nearest_idx = np.argmin(distances, axis=1)

        # Predicted label is label of the nearest training point
        y_pred = self.y_train_[nearest_idx]

        return y_pred

    def score(self, X, y):
        """Compute the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions on X versus y.
        """
        # Validate and also check n_features_in_ against what was seen in fit
        X, y = validate_data(self, X, y, reset=False)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
