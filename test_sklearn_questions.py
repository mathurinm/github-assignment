# ##################################################
# YOU SHOULD NOT TOUCH THIS FILE !
# ##################################################

from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from sklearn_questions import OneNearestNeighbor

from numpy.testing import assert_array_equal


def test_one_nearest_neighbor_check_estimator():
    check_estimator(OneNearestNeighbor())



def test_one_nearest_neighbor_match_sklearn():
    X, y = make_classification(n_samples=200, n_features=20,
                               random_state=42)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
    y_pred_sk = knn.fit(X_train, y_train).predict(X_test)

    onn = OneNearestNeighbor(n_neighbors=1)
    y_pred_me = onn.fit(X_train, y_train).predict(X_test)
    assert_array_equal(y_pred_me, y_pred_sk)

    assert onn.score(X_test, y_test) == knn.score(X_test, y_test)
