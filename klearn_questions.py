[1mdiff --git a/sklearn_questions.py b/sklearn_questions.py[m
[1mindex 75cb93a..44bb4a7 100644[m
[1m--- a/sklearn_questions.py[m
[1m+++ b/sklearn_questions.py[m
[36m@@ -23,8 +23,8 @@[m [mimport numpy as np[m
 from sklearn.base import BaseEstimator[m
 from sklearn.base import ClassifierMixin[m
 from sklearn.utils.validation import check_X_y[m
[32m+[m[32mfrom sklearn.utils.validation import check_array[m
 from sklearn.utils.validation import check_is_fitted[m
[31m-from sklearn.utils.validation import validate_data[m
 from sklearn.utils.multiclass import check_classification_targets[m
 [m
 [m
[36m@@ -71,14 +71,18 @@[m [mclass OneNearestNeighbor(ClassifierMixin, BaseEstimator):[m
             Predicted class labels.[m
         """[m
         check_is_fitted(self)[m
[31m-        X = validate_data(self, X, reset=False)[m
[32m+[m[32m        X = check_array(X)[m
[32m+[m[32m        if X.shape[1] != self.n_features_in_:[m
[32m+[m[32m            raise ValueError([m
[32m+[m[32m                f"X has {X.shape[1]} features, but OneNearestNeighbor "[m
[32m+[m[32m                f"is expecting {self.n_features_in_} features as input."[m
[32m+[m[32m            )[m
         y_pred = np.full([m
             shape=len(X), fill_value=self.classes_[0],[m
             dtype=self.classes_.dtype[m
         )[m
 [m
         for i in range(len(X)):[m
[31m-[m
             distances = np.sqrt(np.sum((self.X_train_ - X[i]) ** 2, axis=1))[m
             nearest_idx = np.argmin(distances)[m
             y_pred[i] = self.y_train_[nearest_idx][m
