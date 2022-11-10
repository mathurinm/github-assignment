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
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the estimator with inputs given.

        It will store both inputs in train_features_ and train_target_
        attributes of the estimator.

        Parameters
        ----------
        X : ndarray of features.
        y : ndarray of classes corresponding to the features given by X.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = len(X[0])

        self.train_features_ = X
        self.n_train_ = len(y)
        self.train_target_ = y
        return self

    def predict(self, X):
        """Predict the class of X.

        Parameters
        ----------
        X : ndarray of features.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        n = len(X)

        for i in range(n):
            prediction = self.train_target_[0]
            dist = np.linalg.norm(X[i] - self.train_features_[0])
            for idx in range(1, self.n_train_):
                test_dist = np.linalg.norm(X[i] - self.train_features_[idx])
                if test_dist < dist:
                    dist = test_dist
                    prediction = self.train_target_[idx]
            y_pred[i] = prediction

        return y_pred

    def score(self, X, y):
        """Evaluate the score of the estimator regarding the inputs.

        It computes the accuracy of the sample, the average good predictions.

        Parameters
        ----------
        X : ndarray of features, from which predictions are made and
        compared with y.
        y : ndarray of classes corresponding to the features given by X, to be
        compared with the predictions of the estimator on X.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        result = 0

        n = len(y)
        for i in range(n):
            if y_pred[i] == y[i]:
                result += 1

        return result/n
