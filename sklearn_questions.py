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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    OneNearestNeighbor classifier.

    Parameters
    ----------
    None

    Attributes
    ----------
    unique_classes_ : ndarray of shape (n_classes,)
        The class labels.
    input_features_count_ : int
        The number of features in the input data.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, data_features, target_values):
        """
        Fit the OneNearestNeighbor model according to the given training data.

        Parameters
        ----------
        data_features : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        target_values : array-like of shape (n_samples)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        data_features, target_values = check_X_y(data_features, target_values)
        check_classification_targets(target_values)
        self.unique_classes_ = np.unique(target_values)
        self.input_features_count_ = data_features.shape[1]

        self.train_features_ = data_features
        self.train_labels_ = target_values

        return self

    def predict(self, input_data):
        """
        Predict the target for the input data.

        Parameters
        ----------
        input_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predicted_labels : ndarray of shape (n_samples,)
            The predicted target.
        """
        check_is_fitted(self)
        input_data = check_array(input_data)
        predicted_labels = np.full(
            shape=len(input_data), fill_value=self.unique_classes_[0],
            dtype=self.unique_classes_.dtype
        )
        for index, data_point in enumerate(input_data):
            distances = np.linalg.norm(self.train_features_ - data_point, axis=1)
            closest_index = np.argmin(distances)
            predicted_labels[index] = self.train_labels_[closest_index]

        return predicted_labels

    def score(self, input_data, true_labels):
        """
        Return the accuracy of the model.

        Parameters
        ----------
        input_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        true_labels : array-like of shape (n_samples)
            The target values.

        Returns
        -------
        accuracy_score : float
            The accuracy of the model.
        """
        input_data, true_labels = check_X_y(input_data, true_labels)
        predicted_labels = self.predict(input_data)

        accuracy_score = np.mean(predicted_labels == true_labels)
        return accuracy_score