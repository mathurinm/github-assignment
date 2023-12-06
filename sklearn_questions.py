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

    Parameters
    ----------
    None

    Attributes
    ----------
    classes_ : array, shape (n_classes,)
        The classes seen during the fit.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score

    >>> # Generate some example data
    >>> X = np.random.rand(100, 2)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)

    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
        )

    >>> # Create and fit the OneNearestNeighbor classifier
    >>> clf = OneNearestNeighbor()
    >>> clf.fit(X_train, y_train)

    >>> # Make predictions on the test set
    >>> y_pred = clf.predict(X_test)

    >>> # Calculate the accuracy
    >>> accuracy = accuracy_score(y_test, y_pred)
    >>> print("Accuracy:", accuracy)
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model.

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Training data input.

        y: array-like, shape (n_samples,)
            Target values.

        Returns:
        --------
        self: object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # The fit method for OneNearestNeighbor is a simple memorization of
        # the training data, as there is no training involved in this model.
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)

        return self

    def predict(self, X):
        """Predict the target values for input data.

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data for prediction.

        Returns:
        --------
        y_pred: array-like, shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # Find the index of the closest point in the training set
        y_pred = [self.y_train_[
            np.argmin(np.linalg.norm(self.X_train_ - x, axis=1))
            ] for x in X]

        return np.array(y_pred)

    def score(self, X, y):
        """
        Compute the accuracy of the model on input data.

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data.

        y: array-like, shape (n_samples,)
            True target values.

        Returns:
        --------
        accuracy: float
            Accuracy of the model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        score = np.mean(y == y_pred)

        return score
