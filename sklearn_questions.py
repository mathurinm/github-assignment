import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.

    This classifier implements a one-nearest neighbor algorithm. For each sample in the test dataset, 
    it predicts the class of the nearest sample in the training dataset based on Euclidean distance.
    """

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Test samples.

        Returns
        -------
        y_pred : array of shape (n_query,)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.array([self._predict_one(x) for x in X])
        return y_pred

    def _predict_one(self, x):
        """
        Predict the class for a single sample.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            A single test sample.

        Returns
        -------
        class_label : The predicted class label for the sample.
        """
        distances = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        nearest_index = np.argmin(distances)
        return self.y_train_[nearest_index]

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
