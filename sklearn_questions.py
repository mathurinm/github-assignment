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
    "OneNearestNeighbor classifier."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the training data to the model

        Parameters
        ----------
        X : numpy.ndarray 
            The training independent variables of the model
        
        y : 1d numpy.ndarray
            The training target vector of the model. 
            (i.e. the classification of the features)

        Returns
        -------
            None

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # XXX fix
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predicts the target variable vector Y for the testing variable
        vector X        
        
        Parameters
        ----------
        X : numpy.ndarray
            The independent variables from the testing set

        Returns
        -------
        y_pred : 1d numpy.ndarray
            The predicted values from the model

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        
        # XXX fix
        predictions = []
        
        for i,x_test in enumerate(X):
             min_dist = np.inf

            
             for j, x_train, y_tr in zip(range(len(self.X_)), self.X_, self.y_):
                dist = np.linalg.norm(x_test - x_train)
                 
                if dist < min_dist:
                    min_dist = dist
                    pred = y_tr
                
             predictions.append(pred)
        
        y_pred = np.array(predictions).flatten()
        return y_pred

    def score(self, X, y):
        """
        Outputs the loss score of the model. Calculated using the 

        Parameters
        ----------
        X : numpy.ndarray
            The matrix containing the independent variables
        y : numpy.ndarray
            The vector containing the target variables

        Returns
        -------
        x : float
            sum of correct predictions
            

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)

        
        return (y_pred==y).sum() / y.shape[0]


