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
    """OneNearestNeighbor classifier.

    Parameters
    ----------
    None

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        The classes found in the training data.
    n_features_in_ : int
        The number of features in the training data.

    Methods
    -------
    fit(X, y)
        Fit the OneNearestNeighbor model to the training data.
    predict(X)
        Predict the class labels for the input data.
    score(X, y)
        Compute the accuracy of the OneNearestNeighbor model on the test data.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model to the training data.

        Parameters
        ----------
        X : array with shape (n_samples, n_features)
            The input training data.
        y : array with shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Store training data for prediction
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the class labels for the input data using the L2 distance.

        Parameters
        ----------
        X : array with shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : array with shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # Compute the L2 distance for each point and store it in a distance
        num_test = X.shape[0]
        num_train = self.X_.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_), axis=1))

        # Find the one nearest neighbour
        for i in range(dists.shape[0]):
            closest_indices = np.argsort(dists[i])[0]
            closest_y = self.y_[closest_indices]
            y_pred[i] = closest_y

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the ONN model on the test data.

        Parameters
        ----------
        X : array with shape (n_samples, n_features)
            The input test data.
        y : array with shape (n_samples,)
            The true labels for the test data.

        Returns
        -------
        accuracy : float
            The accuracy of the model on the test data.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        test = y_pred == y

        return test.sum()/test.shape[0]
