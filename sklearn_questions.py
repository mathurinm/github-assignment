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
        """Attribute properties to the classifier we will use.

        Parameters
        ----------
        self : instance of the OneNearestNeighbor class
            The classifier we will use for our predictions.
        X : numpy array
            The set of features we will train our model on.
        y : numpy array with one column
            The target we train with.

        Returns
        -------
        self : instance of the OneNearestNeighbor class
            The classifier we will use for our predictions.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # XXX fix
        return self

    def predict(self, X):
        """Predict class according to a set of features.

        Parameters
        ----------
        self : instance of the OneNearestNeighbor class
            The classifier we will use for our predictions.
        X : numpy array
            The set of features we predict from.

        Returns
        -------
        y_pred : numpy array
            The predictions.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_ - x, axis=1)
            closest_index = np.argmin(distances)
            y_pred.append(self.y_[closest_index])
        return np.array(y_pred)

    def score(self, X, y):
        """Evaluate the accuracy of the classifier.

        Parameters
        ----------
        self : instance of the OneNearestNeighbor class.
            The classifier we will use for our predictions.
        X : numpy array
            The set of features we predict from.

        Returns
        -------
        accuracy : float
            Proportion of correctly classified samples
        """
        X, y = check_X_y(X, y)
        check_is_fitted(self)
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)

        return accuracy
