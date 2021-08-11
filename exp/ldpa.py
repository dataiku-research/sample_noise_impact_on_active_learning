import openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_config():
    return {
        'start_size': 100,
        'batches': [100] * 29,
        'n_iter': 30,
        'stop_size': 3000
    }

def get_dataset():

    dataset = openml.datasets.get_dataset(1483)                                                                   
    X, y, cat_indicator, names = dataset.get_data(dataset_format='array', target=dataset.default_target_attribute)
    cat_indicator = np.asarray(cat_indicator)

    ct = ColumnTransformer([
        ('encoder', OneHotEncoder(), np.where(cat_indicator)[0]),
        ('normalizer', StandardScaler(), np.where(~cat_indicator)[0])
    ], remainder='passthrough')

    X = ct.fit_transform(X)

    #entropy_per_feature = entropy(X)
    #entropy_mask = np.isfinite(entropy_per_feature)

    #shap = np.load('ldpa_shap.npy')

    return {
        'X': X,
        'y': y
    }


def get_clf():
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1)


def fit_clf(clf, tx, ty, epochs=None):
    return clf.fit(tx, ty)
