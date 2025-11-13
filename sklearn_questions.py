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
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Class labels for training samples.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted classifier.

        Raises
        ------
        ValueError
            If X and y have incompatible shapes or invalid values.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store classes and number of features (sklearn API requirement)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Store training data for nearest‑neighbor lookup
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict class labels for the input samples.

            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
                Input samples.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )

        # For each sample, find the closest training point (1-NN)
        for i in range(X.shape[0]):
            distances = np.linalg.norm(self.X_ - X[i], axis=1)
            # Index of the nearest neighbor
            nearest_idx = np.argmin(distances)
            y_pred[i] = self.y_[nearest_idx]
            
        return y_pred

    def score(self, X, y):
        """Compute the mean accuracy on the given test data and labels.
        
            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
                Test samples.
            y : ndarray of shape (n_samples,)
                True class labels for X.

            Returns
            -------
            score : float
                Mean accuracy of self.predict(X) with respect to y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        return (y_pred == y).mean()