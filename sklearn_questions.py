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
from sklearn.metrics import pairwise_distances_argmin_min


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier - i.e. memorize training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        """
        # Validate and preprocess the input data
        X, y = check_X_y(X, y)
        # Check that the target variable is valid for a classification task
        check_classification_targets(y)
        # Memorize the different classes
        self.classes_ = np.unique(y)
        # Memorize the number of features
        self.n_features_in_ = X.shape[1]
        # Memorize X and y as private attributes
        self._X_fit = X
        self._y_fit = y

        return self

    def predict(self, X):
        """Predict the class labels frolm the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data for prediction
        """
        # Verify that the model was fitted before proceeding to predictions
        check_is_fitted(self)
        # Check that X is a valid array
        X = check_array(X)
        # Predict the labels
        nearest_indices, _ = pairwise_distances_argmin_min(
            X, self._X_fit, metric='euclidean')
        y_pred = self._y_fit[nearest_indices]

        return y_pred

    def score(self, X, y):
        """Return the accuracy of the model on the given data.

        Parameters
        ----------
        X : ndarry of shape (n_samples, n_features)
            Data for testing
        y : ndarray of shape (n_samples,)
            Target vakyes for testing
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy
