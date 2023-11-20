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
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting your model to the training data
            by setting X_train and the y_train.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The features of the training set.

        y : ndarray of shape (n_samples, )
            The observation of the training set.

        Returns
        -------
        self
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
        """Predict the class of a new set of data.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The features of the testing set.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, )
                 Array of the predicted classes
        """

        check_is_fitted(self)
        X = check_array(X)

        # XXX fix
        y_pred = []
        for x in X:
            # Computing the distance between x and x_i
            distances = np.linalg.norm(self.X_train_ - x, axis=1)

            # Computing V(x)
            k_nearest_idx = np.argsort(distances)[0]

            # Get class of k-nearest points
            k_nearest_labels = self.y_train_[k_nearest_idx]

            # Find the most common class and set it as the prediction
            y_pred.append(k_nearest_labels)

        return np.array(y_pred)

    def score(self, X, y):
        """Evaluate de performance of the model by comparing the
           results from our model and the value in the testing set.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The features of the testing set.

        y : ndarray of shape (n_samples, )
            The observation of the testing set.

        Returns
        -------
        score : float
                The number of correct answer devided by the total
                number of answer
        """

        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # XXX fix
        y_compare = y == y_pred
        score = y_compare.sum() / len(y_pred)
        return score
