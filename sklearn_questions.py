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
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(ClassifierMixin, BaseEstimator): # specific parent comes before generic 
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Store the inputs on self: every training point and its label/class.

        Parameters
        ----------
        X : 2D ndarray of shape (n_samples_train, n_features) 
            The design matrix.
        y : 1D ndarray of shape (n_samples,)
            The target.

        Returns
        -------
        self : OneNearestNeighbor
            The fitted classifier. 
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Return the predicted class of the points.
    
        Parameters
        ----------
        X : 2D ndarray of shape (n_samples_test, n_features)
            The samples to classify.
        
        Returns
        -------
        y_pred : 1D ndarray of shape (n_samples_test,) 
            The predicted class label for each sample.  
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneNearestNeighbor "
                f"is expecting {self.n_features_in_} features as input."
            )
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        for i, x in enumerate(X):
            distances = np.sum((self.X_ - x)**2, axis=1) 
            # broadcasting method to subtract the X from every training row; 
            # the subtracted values are then squared and summed
            y_pred[i] = self.y_[np.argmin(distances)] 
            # array of predictions takes the value of of the closest true y value classification to the x point
            # note that we use argmin to retrieve the position of the smallest distance, and not the distance itself

        return y_pred

    def score(self, X, y):
        """Provide an accuracy score of the fitted model.

        Parameters
        ----------
        X : 2D ndarray of shape (n_samples, n_features)
            The samples to classify.
        y : 1D ndarray of shape (n_samples,)
            The true classification of each point.
        
        Returns
        -------
        accuracy : float 
            The proportion of correctly-classified samples. 
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy