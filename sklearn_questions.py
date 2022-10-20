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
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __set_n_features_in(self, X: np.ndarray) -> None:
        """Set 'n_features_in_' attribute, based on input type.

        Parameters
        ----------
        X: np.ndarray
            A training points matrix.
        """
        # Different actions for: 'np.ndarray', 'list', '_NotAnArray' types.
        if hasattr(X, "shape"):
            if len(X.shape) == 1:
                self.n_features_in_ = X.shape[0]
            elif len(X.shape) == 2:
                self.n_features_in_ = X.shape[1]
        elif type(X) == list:
            self.n_features_in_ = np.shape(X)[1]
        else:
            self.__set_n_features_in(X.__array__())

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Perform 'dummy' fit.

        The method just stores training points with related labels.

        Parameters
        ----------
        X: np.ndarray
            Training points.

        y: np.ndarray
            Training points' labels.

        Returns
        -------
        self: BaseEstimator
            An object itself.
        """
        # Checks.
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Set necessary attributes.
        self.X_ = X
        self.y_ = y
        self.__set_n_features_in(X)
        self.classes_ = unique_labels(y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict points' labels based on the nearest train point's label.

        Parameters
        ----------
        X: np.ndarray
            Input points.

        Returns
        -------
        y_pred: np.ndarray
            Predicted labels.
        """
        # Checks.
        check_is_fitted(self)
        X = check_array(X)

        # Perform labels prediction.
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        predictions = self.y_[closest]

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> float:
        """Get an accuracy of an estimator.

        Parameters
        ----------
        X: np.ndarray
            Input points.
        y: np.ndarray
            Input points' labels (ground truth).

        Returns
        -------
        accuracy: float
            Resulted accuracy of an estimator.
        """
        # Checks.
        X, y = check_X_y(X, y)

        # Perform accuracy calculation.
        y_pred = self.predict(X)
        accuracy = sum(y_pred == y) / len(y)

        return accuracy
