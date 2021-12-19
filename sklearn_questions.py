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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model on data.

        Parameters
        ----------
        X : np.ndarray
            feature matrix
        y : np.ndarray
            array of labels
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # init fit model
        self._model = KNeighborsClassifier(n_neighbors=1)
        self._model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make a prediction.

        Parameters
        ----------
        X : np.ndarray
            matrix of features
        Returns
        -------
        np.ndarray
            returns an array of predictions
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        model = self._model
        # predict
        y_pred = model.predict(X)

        return y_pred

    def score(self, X: np.array, y: np.array) -> float:
        """Assess the model.

        Parameters
        ----------
        X : np.array
            matrix of features
        y : np.array
            array of true labels
        Returns
        -------
        float
            returns the score of the model
        """
        X, y = check_X_y(X, y)

        model = self._model

        score = model.score(X, y)
        return score
