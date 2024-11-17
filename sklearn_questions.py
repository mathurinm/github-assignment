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
        """We fit the OneNearestNeighbor estimator on the training data.

        Parameters
        ----------
        X : ndarray, a list or a sparse matrix
        y : ndarray, a list or a sparse matrix

        Returns
        ----------
        self : the instance of the class on which it was called.
            Also, attributes are created :
            .classes_ will return the number of unique values (i.e. the
            classes) contained in y
            .n_features_in_ will return the number of columns (i.e. the
            features) contained in X
            ._X_train will return X
            ._y_train will return y

        Raises
        ------
        ValueError
            If X and y are not of consistent length or
            if y is of regression type.
        """
        # We do some checks:
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # We store the classes (unique values of y) and number of features:
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # We save the training data for prediction:
        self._X_train = X
        self._y_train = y

        return self

    def predict(self, X):
        """We do the prediction of the OneNearestNeighbor estimator.

        Parameters
        ----------
        X : a 2D array of shape (n_samples, n_features_in_), on which we will
            test the data to classify

        Returns
        ----------
        y_pred : a 1D array of shape (n_samples,) on which the data has been
            fitted.

        Raises
        ------
        ValueError
            If X is not of correctly fitted with the training data or
            if X is not of correct shape (i.e. not of dimension
            (n_samples, n_features_in_))
        """
        # We do some checks:
        check_is_fitted(self)
        X = check_array(X)

        # We initiate the y_pred matrix that will contain the predictions:
        y_pred = np.full(
            shape=len(X),
            fill_value=self.classes_[0],
            dtype=self.classes_.dtype,
        )

        for i, X_test in enumerate(X):
            euclidian_dist = np.linalg.norm(self._X_train - X_test, axis=1)
            nearest_index = np.argmin(euclidian_dist)
            y_pred[i] = self._y_train[nearest_index]
        return y_pred

    def score(self, X, y):
        """We evaluate the score of our prediction.

        We compare the accurate vs. non-accurate prediction that we got on the
        test set after having trained the OneNearestNeighbor on the training
        set.

        Parameters
        ----------
        X : ndarray, a list or a sparse matrix
        y : ndarray, a list or a sparse matrix

        Returns
        ----------
        accuracy score : a float nimber computed as the ratio of correct
            predictions over the total number of predictions that have been
            made.

        Raises
        ------
        ValueError
            If X and y are not of consistent length
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # We create a Boolean array that highlight the correct/non-correct
        # predictions of y_pred:
        correct_pred = (y_pred == y).astype(int).sum()

        # We compute the accuracy score:
        accuracy_score = correct_pred / len(y)

        return accuracy_score
