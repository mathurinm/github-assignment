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
        """
        Group the points of X depending on their target.

        Input:
        X : 2D array whose first dimension is the number of points, the second is the number of features
        y : 1D array with the associated tagets

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.data_ = (X,y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Give the prediction associated to the training point with the smallest euclidian distance.

        Output:
        y : 1D array containing the preedictions


        Input:
        Fitted model (self)
        X : 2D array whose first dimension is the number of points, the second is the number of features
        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        train, targets = self.data_


        for i in range(len(X)):
            x = X[i]
            normes = np.linalg.norm(train - x, axis=1)

            k = np.min(np.argmin(normes))

            y_pred[i] = targets[k]

        return y_pred


    def score(self, X, y):
        """
        Compare line by line the difference between the prediction and the actual values, and calculates the average.

        Input:
        X : 2D array whose first dimension is the number of points, the second is the number of features
        y : 1D array with the associated taget
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        y_pred = np.equal(y_pred, y)
        n = len(y_pred)
        # XXX fix
        return (y_pred.sum()/n)

