import pickle

import bson
import librosa.display
import numpy as np
import pandas as pd
import pymongo
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split

from preprocessing import get_tracks_from_db

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def build_and_compile_model(batch_size, train_features):
    normalization = tf.keras.layers.Normalization(axis=-1)
    normalization.adapt(np.array(train_features))

    model = keras.Sequential([
        layers.InputLayer(input_shape=(*features[feature]['shape'], 1)),
        normalization,
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPooling2D(padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.Flatten(name='Flattening'),
        layers.Dense(32),
        layers.Dense(16),
        layers.Dense(8),
        layers.Dense(1, name='Prediction')
    ])

    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    model.build(input_shape=(batch_size, *features[feature]['shape'], 1))
    return model


def print_target_stats(test_or_training, target):
    print(f"####### {test_or_training} data stats #########")
    print(f"min streams: {target.min()}")
    print(f"max streams: {target.max()}")
    print(f"avg streams: {target.mean()}")
    print(f"median streams: {target.median()}")


def plot_loss(history, feature):
    y_max = max(max(history.history['loss']), max(history.history['val_loss']))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, y_max + y_max*0.1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.title(f"Training on {feature}")

    plt.show()


def reshape_feature(feat: np.array):
    try:
        return feat.reshape(*features[feature]['shape'], 1)
    except Exception as e:
        print(f"Could not reshape array of shape {feat.shape}")
        print(e)


def train_model(dataset, feature):
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    train_features = train_dataset[feature]
    test_features = test_dataset[feature]

    train_targets = train_dataset.pop('streams')
    test_targets = test_dataset.pop('streams')

    print_target_stats('training', train_targets)
    print_target_stats('test', test_targets)

    train_targets = train_targets.to_numpy()
    test_targets = test_targets.to_numpy()

    train_features = train_features.to_numpy() #.reshape(-1, 128, 1292, 1)
    test_features = test_features.to_numpy() #.reshape(-1, 128, 1292, 1)

    train_targets = train_targets.reshape(len(train_targets), 1) #[[a] for a in train_targets]
    test_targets = test_targets.reshape(len(test_targets), 1) #[[a] for a in test_targets]

    train_features = np.array([reshape_feature(a) for a in train_features])
    test_features = np.array([reshape_feature(a) for a in test_features])

    BATCH_SIZE = 4

    model = build_and_compile_model(BATCH_SIZE, train_features)

    print(model.summary())

    history = model.fit(
        train_features,
        train_targets,
        epochs=40,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.2)

    plot_loss(history, feature)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail(20))
    print()

    test_results = {}
    test_results['model'] = model.evaluate(
        test_features, test_targets, verbose=1)

    print(test_results['model'])
    print()

    print(model.predict(train_features[:1]))
    print(train_targets[:1])


if __name__ == "__main__":

    features = {
        'tonnetz': {
            'shape': (6, 323)
        },
        'melspec': {
            'shape': (128, 323)
        }
    }

    feature = "melspec"

    tracks = get_tracks_from_db(feature).limit(6000)

    df = pd.DataFrame(tracks)

    df = df[['streams', feature]]

    df = df.dropna()

    df['streams'] = pd.to_numeric(df['streams'], downcast='integer')

    def unpack_array(pickled_array):
        arr = pickle.loads(pickled_array)
        if type(arr) == np.ndarray and arr.shape == features[feature]['shape']:
            return arr
        return np.nan

    df[feature] = df[feature].map(unpack_array)

    df.dropna(inplace=True)

    train_model(df, feature)
