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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest-neighbor classifier.

    This classifier predicts, for each sample, the label of the closest
    training sample, using the Euclidean distance.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    X_ : ndarray of shape (n_samples, n_features)
        Training data stored after fitting.

    y_ : ndarray of shape (n_samples,)
        Target values stored after fitting.

    n_features_in_ : int
        Number of features seen during fit.
    """

    def fit(self, X, y):
        """Fit the one-nearest-neighbor classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, y)

        self._knn = knn
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ("_knn", "classes_", "n_features_in_"))
        X = check_array(X, accept_sparse=False)
        return self._knn.predict(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of the predictions on the given data.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
