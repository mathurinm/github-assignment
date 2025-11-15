"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.

The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the fit,
predict and score methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo pytest test_sklearn_questions.py.

We also ask to respect the pep8 convention: https://pep8.org. This will be
enforced with flake8. You can check that there is no flake8 errors by
calling flake8 at the root of the repo.

Finally, you need to write docstring similar to the one in numpy_questions
for the methods you code and for the class. The docstring will be checked using
pydocstyle that you can also call at the root of the repo.
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

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        This method saves the training samples and their labels,
        used to predict the label of a test sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. Each row represents one sample, and each column
            represents a feature.

        y : ndarray of shape (n_samples,)
            Target labels corresponding to the training samples. Must be valid
            classification targets (e.g., integers or strings).

        Returns
        -------
        self : OneNearestNeighbor
            The fitted OneNearestNeighbor classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for the given samples.

        For each input sample, the label of the closest training sample
        is returned based on Euclidean distance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data, where each row corresponds to a sample and each column
            corresponds to a feature.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each input sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        euc_dist = np.sqrt(((X[None, :, :] - self.X_[:, None, :])**2)
                           .sum(axis=2))
        nearest_point = np.argmin(euc_dist, axis=0)
        y_pred = self.y_[nearest_point]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier on test data.

        This method predicts the labels and compares
        them to the true labels
        to calculate the fraction of correctly classified samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples, row represents a sample and column a feature.

        y : ndarray of shape (n_samples,)
            True labels corresponding to the test samples.

        Returns
        -------
        accuracy : float
            Proportion of samples that were correctly classified.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
