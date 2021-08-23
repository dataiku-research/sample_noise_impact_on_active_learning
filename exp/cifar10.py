import openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_config():
    return {
        'start_size': 1000,
        'batches': [1000] * 9,
        'n_iter': 10,
        'stop_size': 10000,
        'full_dataset_fit_params': {'epochs': 30}
    }
# from sklearn.metrics import silhouette_score, calinski_harabaz_score



def get_dataset():
    X = np.load('cifar_embeddings.npy')
    y = np.load('cifar_target.npy')

    return {
        'X': X,
        'y': y
    }


def get_clf():
    import keras
    from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
    from tensorflow.keras import Model
    from tensorflow.keras import optimizers, layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical

    keras.backend.clear_session()

    model = Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    return model


def fit_clf(clf, tx, ty, epochs=10):
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    return clf.fit(tx, ty, epochs=epochs, callbacks=[early_stopping_monitor])
