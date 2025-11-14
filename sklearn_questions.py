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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest-neighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        This method stores the training data X and y inside the estimator
        so that predictions can be made based on the nearest neighbor rule.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels corresponding to X.

        Returns
        -------
        self : object
            Fitted estimator.
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

        For each sample in X, this method finds the closest training sample
        stored during ``fit`` using the Euclidean distance, and returns its
        corresponding label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict class labels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample in X.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneNearestNeighbor "
                f"is expecting {self.n_features_in_} features as input."
            )

        y_pred = np.full(
            shape=len(X),
            fill_value=self.classes_[0],
            dtype=self.classes_.dtype,
        )

        for idx, x_i in enumerate(X):
            diff = self.X_ - x_i
            distances = np.sqrt(np.sum(diff**2, axis=1))
            nearest = np.argmin(distances)
            y_pred[idx] = self.y_[nearest]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        This method compares the predicted labels for X with the true labels y
        and returns the proportion of correctly classified samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        accuracy : float
            Mean accuracy of the classifier on the given test data.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        correct = y_pred == y
        accuracy = np.mean(correct)
        return accuracy
