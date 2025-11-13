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
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate X and y
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store training data
        self.X_train_ = X
        self.y_train_ = y

        # Required by sklearn API
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict class labels using nearest neighbor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self)

        X = check_array(X)

        # required by sklearn test: MUST match regex
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"OneNearestNeighbor is expecting {self.n_features_in_} features as input"
            )

        y_pred = []

        for x in X:
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            idx = np.argmin(distances)
            y_pred.append(self.y_train_[idx])

        return np.array(y_pred)


    def score(self, X, y):
        """Compute accuracy score.

        Parameters
        ----------
        X : array-like
            Input features.

        y : array-like
            True labels.

        Returns
        -------
        accuracy : float
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
