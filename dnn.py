import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


def build_and_compile_model(num_features, train_features):

    normalization = tf.keras.layers.Normalization(axis=-1)
    normalization.adapt(np.array(train_features))

    model = keras.Sequential([
        normalization,
        layers.Dense(num_features, activation='relu'),
        layers.Dense(num_features/2, activation='relu'),
        layers.Dense(num_features/3, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(0.01))
    model.build(input_shape=(None, num_features))
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
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_targets = train_features.pop('streams')
    test_targets = test_features.pop('streams')

    print_target_stats('training', train_targets)
    print_target_stats('test', test_targets)

    BATCH_SIZE = 128
    NUM_FEATURES = train_features.shape[1]

    model = build_and_compile_model(NUM_FEATURES, train_features)

    print(model.summary())

    print(model.predict(train_features[:1]))

    history = model.fit(
        train_features,
        train_targets,
        epochs=200,
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


def create_dataset_with_random_streams():
    from random import randrange

    num_rows = 10000
    d = {
        'feat1': [1 for i in range(num_rows)],
        'feat2': [1for i in range(num_rows)],
        'feat3': [1 for i in range(num_rows)],
        'streams': [randrange(100) for i in range(num_rows)]
    }
    return pd.DataFrame(data=d)


if __name__ == "__main__":

    features = ['mel_spectrogram', 'tonnetz', 'chromagram', 'mel_spec-tonnetz-chromagram']

    for feature in features:
        df = pd.read_csv(f"data/{feature}.csv")[:]
        df.drop(['track_id', 'Unnamed: 0'], axis=1, inplace=True)

        train_model(df, feature)
