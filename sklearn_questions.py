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
from sklearn import metrics
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
        """X and y are the training data 
        that are going to be used as a reference
        to assign values in prediction
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        # Storing data :
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        
        return self

    def predict(self, X):
        """we compute pariwise distances between the stored reference 
        samples and the test samples before assigning as prediction the
        value of the nearest neighbour
        """
        check_is_fitted(self)
        X = check_array(X)
        # Compute all pairwise distances between X and self.X_
        M = metrics.pairwise.pairwise_distances(
            X, Y=self.X_, metric='euclidean', n_jobs=1)

        # Get indices to sort them by distance
        indices = np.argsort(M, axis=1)

        # Get the indexs of the nearest neighbours for each sample.
        neighbors = indices[:, 0]

        # Get labels of each nearest neighbour, which is y_pred
        y_pred = self.y_[neighbors]

        return y_pred

    def score(self, X, y):
        """We predict samples and score the prediction using
        the testing target values
        """

        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return (y_pred==y).mean()