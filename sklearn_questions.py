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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the nearest classifier

        Args:
            X (ndarray): ndarray representing the data in 2D.
            y (ndarray): target values. 1D.

        Returns:
            _self_ : an object with several attributes : classes, X_train, y_train, n.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train = X
        self.y_train = y
        self.n = X.shape[1]
        
        return self

    def predict(self, X):
        """_summary_

        Args:
            X (2D ndarray): 

        Returns:
            _type_: _description_
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        
        norm = euclidean_distances(X, self.X_train)
        index = np.argmin(norm, axis = 1)

        return self.y_train[index]

    def score(self, X, y):
        """Write docstring.

        And describe parameters
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        
        #The original KNeighborsClassifier is a subclass of the 
        # sklearn.base.ClassifierMixin. 
        # The score method uses accuracy_score. 
        # To match the results, we use the same method
        return accuracy_score(y_pred, y)
