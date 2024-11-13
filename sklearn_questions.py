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
        """Check X, y, stores them for future usage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            n x p matrix with n training data points and p features.

        y : array of shape (n_samples x 1)
            Vector with n outcome variables

        Returns
        -------
        self : OneNearestNeighbour
            The fitted OneNearestNeighbour
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predicts using One Nearest Neighbour for each observation in X.
        For each observation in X, we find the closest point,
        in the fitted data and return its label.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

            n x p matrix with n data points and p features,
            on which we predict.

        Returns
        -------
        y_pred : array of shape (n_samples x 1)

            The predicted outcomes.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i in range(len(y_pred)):

            distances = []
            for j in range(len(self.X_)):
                dist = np.linalg.norm(X[i]-self.X_[j])
                distances.append(dist)

            y_pred[i] = self.y_[np.argmin(distances)]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy on the given X and y.
        Calls predict and compares the labels

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            n x p matrix with n test data points and p features.

        y : array of shape (n_samples x 1)
            Test vector with n outcome variables, to be compared with
            model's predictions on X

        Returns
        -------
        score : float
            The accuracy based on y and predictions of X.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        result = []
        for i in range(len(y_pred)):
            if y_pred[i] == y[i]:
                result.append(1)
            else:
                result.append(0)

        score = np.sum(result) / len(y_pred)

        return score
