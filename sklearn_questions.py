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


def most_common(lst):
    """Return most common element in list."""
    return max(set(lst), key=lst.count)


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fits the model.

        Parameters
        self - instance of One neirest neighbor
        X - Features of the training set
        y - Labels of the training set

        Returns
        self - updated instance of One neirest neighbor
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Predict labels for new features.

        Parameters
        self - instance of One neirest neighbor
        X - Features of the test set

        Returns
        np.array(y_pred) - numpy array with estimated labels
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        neighbors = []
        for x in X:
            distances = np.sqrt(np.sum((x - self.X_train_)**2, axis=1))
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train_))]
            neighbors.append(y_sorted[:1])
        y_pred = list(map(most_common, neighbors))
        return np.array(y_pred)

    def score(self, X, y):
        """
        Evaluate the model.

        Parameters
        self - instance of One neirest neighbor
        X - Features of the test set
        y - Labels of the test set

        Returns
        accuracy - score to evaluate error of the model
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        accuracy = sum(y_pred == y) / len(y)

        return accuracy
