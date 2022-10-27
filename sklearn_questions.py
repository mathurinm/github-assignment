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
from sklearn.metrics import euclidean_distances, accuracy_score


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring.
        
        Fit the one-nearest neighbor classifier.
        Parameters
        ----------
        X : array like, of shape n_samples and n_features
            Training data (X_train).
        y : array like, of shape n_samples and 1
            target values (y_train).

        Returns
        -------
        self : the classifier itself once fitted.

        Errors
        -------
        ValueError : if X and y don't match size-wise, \
        if y is not of a good format.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # XXX fix
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Write docstring.

        Predict the y for the provided data.
        Parameters
        ----------
        X : array like, of shape n_samples and n_features
            Testing data (X_test).

        Returns
        -------
        y : array like, of shape n_samples and 1
            predicted values for each test data (y_pred).
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # XXX fix
        distance = euclidean_distances(X, self.X_train_)
        shorter_distance = np.argmin(distance, axis=1)
        return self.y_train_[shorter_distance]

    def score(self, X, y):
        """Write docstring.

        Returns the score of a predicted y compared to the real values.

        Parameters
        ----------
        X : array like, of shape n_samples and n_features
            Imput data (X_test).
        y : array like, of shape n_samples and 1
            True values for the imput data (y_test).

        Returns
        -------
        Result: the accuracy score of the y_pred against the true y.

        Errors
        -------
        ValueError : if X and y don't match size-wise.

        And describe parameters
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        result = accuracy_score(y_pred, y)
        return result
