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
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.

    Parameters
    ----------
    None

    Attributes
    ----------
    classes_ : array, shape (n_classes,)
        The unique classes present in the training data.

    X_ : array-like or pd.DataFrame, shape (n_samples, n_features)
        The input data stored during fitting.

    y_ : array-like, shape (n_samples,)
        The target values stored during fitting.

    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor model.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict the target values for input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self)
        X = check_array(X)

        distances = cdist(X, self.X_, metric='euclidean')
        nearest_indices = np.argmin(distances, axis=1)
        y_pred = self.y_[nearest_indices]

        return y_pred

    def score(self, X, y):
        """
        Returns the accuracy of the model on the given data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        accuracy : float
            The accuracy of the model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        #accuracy = accuracy_score(y, y_pred)

        return np.mean(y_pred == y)
