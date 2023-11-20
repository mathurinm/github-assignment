import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass


    def fit(self, X, y):
        """Fit the OneNearestNeighbor classifier.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Training data.
        y : array-like or pd.Series, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # Set the n_features_in_ attribute
        self.n_features_in_ = X.shape[1]
    

        # Calculate and store any necessary information for prediction
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict the target values for input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Data for which to predict the target values.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.empty(len(X), dtype=self.classes_.dtype)

        for i, x_test in enumerate(X):
            # Find the index of the closest training example to x_test
            closest_index = np.argmin(np.linalg.norm(x_test - self.X_train_, axis=1))
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

        # Calculate mean accuracy
        accuracy = np.mean(y_pred == y)

        return accuracy
