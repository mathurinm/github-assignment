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
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import validate_data
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(
            self,
            X, y,
            ensure_2d=True,
            dtype=None
            )
        
        check_classification_targets(y)
        
        self.X_ = X
        self.y_ = y

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the label of each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples for which we want to guess the label. They must have the 
            same number of features as the data used during ``fit``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels. For each sample, the model looks for the closest
            point in the training set and returns its label.
        """

        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=None,
            reset=False
        )

        distance = np.linalg.norm(self.X_[None,:,:] - X[:,None,:], axis=2)
        nearest_index = np.argmin(distance, axis=1)
        y_pred = self.y_[nearest_index]
        
        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples used to evaluate the model.

        y : array-like of shape (n_samples,)
            True labels corresponding to X.

        Returns
        -------
        score : float
            The accuracy of the predictions, between 0 and 1.
        """
        X, y = validate_data(
            self,
            X, y,
            ensure_2d=True,
            dtype=None,
            reset=False
        )
        y_pred = self.predict(X)


        return (y_pred == y).mean()
