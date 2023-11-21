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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from scipy.spatial.distance import cdist


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.
    Parameters
    ----------
    No parameters are required for initialization.
    Attributes
    ----------
    classes_ : array-like
        The unique classes found in the training data.
    Methods
    -------
    fit(X, y)
        Fit the OneNearestNeighbor classifier to the training data.
    predict(X)
        Predict the class labels for input samples.
    score(X, y)
        Return the mean accuracy on the given test data and labels.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier to the training data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input training samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self for chaining.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the class labels for input samples.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels.

        """
        check_is_fitted(self)

        # Check whether X is a non empty array
        X = check_array(X)

        # Euclidean distances between new data X and training data self.X_.
        distances = cdist(X, self.X_, metric="euclidean")

        # Store the index of the nearest neighbor for each new point.
        nn_index = distances.argmin(axis=1)

        return self.y_[nn_index]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input test samples.
        y : array-like, shape (n_samples,)
            The true labels.

        Returns
        -------
        accuracy : float
            Mean accuracy of the classifier on the test data.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # Return the Mean accuracy of the classifier
        return np.mean(y_pred == y)
