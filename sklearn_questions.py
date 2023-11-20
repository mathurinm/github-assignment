"""
sklearn_questions.py

This module contains the implementation of the OneNearestNeighbor classifier for scikit-learn.

- The OneNearestNeighbor class is a custom scikit-learn estimator for the nearest neighbor classifier.
- It includes the fit, predict, and score methods.

For usage instructions and examples, refer to the docstrings within the class and methods.
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
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    >>> # Create and fit the OneNearestNeighbor classifier
    >>> clf = OneNearestNeighbor()
    >>> clf.fit(X_train, y_train)

    >>> # Make predictions on the test set
    >>> y_pred = clf.predict(X_test)

    >>> # Calculate the accuracy
    >>> accuracy = accuracy_score(y_test, y_pred)
    >>> print("Accuracy:", accuracy)
    """

    def __init__(self):
        """Initialize the OneNearestNeighbor classifier."""
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The training input samples.
        y : array-like or pd.Series, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # The fit method for OneNearestNeighbor is a simple memorization of
        # the training data, as there is no training involved in this model.
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict the target values for input samples.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        for i, x_i in enumerate(X):
            # Find the index of the closest point in the training set
            closest_index = np.argmin(np.linalg.norm(x_i - self.X_train_,
                                                     axis=1))
            y_pred[i] = self.y_train_[closest_index]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Test samples.
        y : array-like or pd.Series, shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
