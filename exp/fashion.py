import openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier


def get_config():
    return {
        'start_size': 100,
        'batches': [200] * 24,
        'n_iter': 25,
        'stop_size': 4900
    }

def get_dataset():
    dataset = openml.datasets.get_dataset(40996)
    X, y, cat_indicator, a = dataset.get_data(dataset_format='array', target=dataset.default_target_attribute)
    X = X.astype('float32') / 255.

    return {
        'X': X,
        'y': y
    }


def get_clf():
    return MLPClassifier(hidden_layer_sizes=(128, 64))


def fit_clf(clf, tx, ty):
    return clf.fit(tx, ty)
