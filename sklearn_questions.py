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
from sklearn.utils.validation import validate_data


class OneNearestNeighbor(ClassifierMixin, BaseEstimator):
    """One-nearest-neighbor classifier using the Euclidean distance.

    This estimator implements a simple 1-nearest-neighbor classifier. For each
    input sample, the predicted label is the target of the closest training
    sample according to the Euclidean-2 distance.

    The estimator follows the scikit-learn interface and can therefore be used
    in pipelines and with utilities such as ``check_estimator``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    X_train_ : ndarray of shape (n_samples, n_features)
        Training data used during :meth:`fit`.

    y_train_ : ndarray of shape (n_samples,)
        Target values corresponding to ``X_train_``.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the one-nearest-neighbor classifier.

        This stores the training samples and their corresponding targets.
        Input validation is performed and the number of features is recorded
        in the ``n_features_in_`` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target values corresponding to the rows of ``X``. The targets are
            expected to be compatible with a classification task.

        Returns
        -------
        self : OneNearestNeighbor
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have inconsistent shapes or if the targets are
            not valid for a classification problem.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predict class labels for the given samples.

        For each sample in ``X``, the label of the closest training sample
        (in Euclidean distance) is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict class labels. The number of
            features must match the number of features seen during ``fit``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample in ``X``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.

        ValueError
            If the number of features in ``X`` is different from the number
            of features seen during ``fit`` or if the input cannot be
            validated.
        """
        check_is_fitted(self)
        X = check_array(X)

        # "validate_data" validates the input data X, for example, that the
        # number of features are the same as the number of features of
        # the fitted model
        X = validate_data(self,
                          X,
                          ensure_2d=True,
                          dtype=None,
                          reset=False)

        # uses broadcasting to compute the difference between all points of the
        # X_fitted data and the X data we want to predict
        X_diff = self.X_train_.T[:, None, :] - X.T[:, :, None]

        # computes the linalg norm 2
        res = np.linalg.norm(X_diff, ord=2, axis=0)

        # for each column we use unravel as in the numpy_questions.py file
        # to get the index of the point in X_train_ fit the smallest distance
        indexes = np.unravel_index(
            np.argmin(res, axis=1), res.shape)[1]

        # init the y_pred model, specifically the predicted values for the
        # classification by taking the closests point index
        y_pred = np.full(
            shape=len(X), fill_value=self.y_train_[indexes],
            dtype=self.classes_.dtype
        )

        return y_pred

    def score(self, X, y):
        """Compute the mean accuracy on the given test data and labels.

        This is the fraction of correctly classified samples, i.e. the average
        of ``y_pred == y`` where ``y_pred`` is obtained from :meth:`predict`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for ``X``.

        Returns
        -------
        score : float
            Mean accuracy of the classifier on the provided data, in the
            interval [0.0, 1.0].

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have inconsistent shapes or if the input cannot
            be validated.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        # computes the accuracy
        # number of correct predictions / all predictions
        return (y_pred == y).sum() / len(y_pred)
