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

    Attributes
    ----------
    classes_ : array, shape (n_classes,)
        The classes seen during fitting.
    n_features_in_ : int
        The number of features in the input data.

    Methods
    -------
    fit(X, y)
        Fit the OneNearestNeighbor classifier to the training data.

    predict(X)
        Predict the class labels for the input data.

    score(X, y)
        Compute the accuracy of the classifier on the input data.

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier to the training data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Training data.
        y : array-like or pd.Series, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict the values for the input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data for which to predict class labels.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values for X.

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), 
                         fill_value=self.classes_[0], 
                         dtype=self.classes_.dtype)

        for i, x_i in enumerate(X):
            closest_index = np.argmin(
                np.linalg.norm(x_i - self.X_train_, axis=1))
            y_pred[i] = self.y_train_[closest_index]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier on the input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : array-like or pd.Series, shape (n_samples,)
            True labels for input data.

        Returns
        -------
        accuracy : float
            Accuracy of the classifier on the input data.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        accuracy = np.mean(y_pred == y)

        return accuracy
