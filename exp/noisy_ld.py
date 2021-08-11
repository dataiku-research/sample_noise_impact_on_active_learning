import openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_config():
    return {
        'start_size': 20,
        'batches': [10] * 19,
        'n_iter': 20,
        'stop_size': 220,
        #'oracle_error': .1
    }

def get_dataset():
    X = np.load('X.npy')
    y = np.load('y.npy')

    return {
        'X': X,
        'y': y
    }


def get_clf():
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1)


def fit_clf(clf, tx, ty):
    return clf.fit(tx, ty)
