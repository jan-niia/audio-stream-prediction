import pickle

import bson
import numpy as np
import pandas as pd
import pymongo
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split

from preprocessing import get_tracks_from_db


def build_and_compile_model(batch_size):

    # normalization = tf.keras.layers.Normalization(axis=-1)
    # normalization.adapt(train_features)

    model = keras.Sequential([
        #layers.InputLayer(input_shape=(batch_size, 128, 1292, 1)),
        #normalization,
        layers.Conv2D(32, (4, 4), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPooling2D(),
        layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(0.001))
    model.build(input_shape=(batch_size, 128, 1292, 1))
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


def train_model(dataset, feature):
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    train_features = train_dataset['melspec']
    test_features = test_dataset['melspec']

    train_targets = train_dataset.pop('streams')
    test_targets = test_dataset.pop('streams')

    print_target_stats('training', train_targets)
    print_target_stats('test', test_targets)

    BATCH_SIZE = 32

    model = build_and_compile_model(BATCH_SIZE)

    print(model.summary())

    #try to predict first element of training data before model is trained

    # train_features = train_features.reshape(1, 128, 1292, 1)
    # test_features = test_features.reshape(1, 128, 1292, 1)

    #print(model.predict(tensor))
    print(model(train_features.iloc[0].reshape(1, 128, 1292, 1)))


    train_features = train_features.to_numpy()

    print(type(train_features))
    print(type(train_features[0]))
    print(type(train_features[0][0]))
    print(type(train_features[0][0][0]))


    test_features = test_features.to_numpy()


    train_features = np.column_stack(train_features)
    test_features = np.column_stack(test_features)


    print(train_features)

    # train_features.reshape(128, 1292, 1)
    # test_features.reshape(128, 1292, 1)

    history = model.fit(
        train_features,
        train_targets,
        epochs=10,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.2)

    plot_loss(history, feature)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail(20))

    test_results = {}
    test_results['model'] = model.evaluate(
        test_features, test_targets, verbose=1)

    print(test_results['model'])

    print(model.predict(train_features[:1]))


if __name__ == "__main__":

    features = ['mel_spectrogram', 'mel_spec-tonnetz-chromagram']#, 'tonnetz', 'chromagram', ]

    tracks = get_tracks_from_db().limit(1000)
    #
    # tracks_list = list(tracks)
    #
    # print(tracks_list[0])
    #
    # tracks_list = [pickle.loads(t['melspec']) for t in tracks_list]

    df = pd.DataFrame(tracks)

    df = df[['streams', 'melspec']]

    df = df.dropna()

    df['melspec'] = df['melspec'].map(lambda x: pickle.loads(x))

    train_model(df, 'mel_spectrogram')
