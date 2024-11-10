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


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit model to the data and ensure it's correctly fitted.

        Parameters
        ----------
        self : has been defined precedently as a constructor with no parameters
        X : ndarray of dim = 2d, corresponds to the data used here
        check done ensuring it has the correct format
        y : ndarray of dim = 1d, correspond to the target we aim for
        check done with check_classification to ensure it is
        filled with "correct targets"

        Returns
        -------
        self : the model fitted to the data

        Other functions used
        --------------------
        self.classes gives us the different classes in y
        self.n_features_in_ gives us the nb of features in our data X
        self.X_train_ and self.Y_train enable us to store X and y
        while fitting them with the model
        we will use them afterwards for predictions
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.Y_train_ = y
        return self

    def predict(self, X):
        """Compute the euclidian distance for each point.

        Parameters
        ----------
        self : our model, fitted in the above function
        with our data X and y
        check done to ensure the model has been correctly fitted
        X : ndarray of dim = 2d, corresponds to the data used here
        check done ensuring it has the correct format

        Returns
        -------
        y_pred : the prediction we make according to our model
        using the data X
        To fill y_pred with thecorrect values:
        * create an empty array
        * compute the euclidian distance pairwise for our data
        ie comparing X and X_train, that we trained following our model
        * filling y_pred with, for each Y_train :
        the narrowest euclidian distance found and stored in distances
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        distances = euclidean_distances(X, self.X_train_)
        for i in range(len(X)):
            index = np.argmin(distances[i])
            y_pred[i] = self.Y_train_[index]
        return y_pred

    def score(self, X, y):
        """Compare predictions with true values expected.

        Parameters
        ----------
        self : our model, fitted in the above function with our data X and y
        X : ndarray of dim = 2d, corresponds to the data used here
        check done ensuring it has the correct format
        y : ndarray of dim = 1d, the target
        check done ensuring it has the correct format

        Returns
        -------
        The % of "good predictions" in our model by taking
        the mean of Booleans in y_pred == y
        ie the nb of times y_pred and y are the same
        divided by the nb of comparisons made
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
