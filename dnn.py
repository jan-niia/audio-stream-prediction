import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


def build_and_compile_model(num_features):
    model = keras.Sequential([
        layers.Dense(num_features, activation='relu'),
        layers.Dense(num_features/2, activation='relu'),
        layers.Dense(num_features/3, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(0.01))
    model.build(input_shape=(None, num_features))
    return model


def plot_loss(history):
    y_max = max(max(history.history['loss']), max(history.history['val_loss']))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, y_max + y_max*0.1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.show()


def train_model(dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('streams')
    test_labels = test_features.pop('streams')

    BATCH_SIZE = 256
    NUM_FEATURES = train_features.shape[1]

    model = build_and_compile_model(NUM_FEATURES)

    print(model.summary())

    history = model.fit(
        train_features,
        train_labels,
        epochs=100,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.2)

    plot_loss(history)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail(20))

    print(type(history.history['val_loss']))
    print(history.history['val_loss'])

    test_results = {}
    test_results['model'] = model.evaluate(
        test_features, test_labels, verbose=1)

    print(test_results['model'])


if __name__ == "__main__":
    df = pd.read_csv('data/mel_spectrogram.csv')[:10000]
    df.drop(['track_id', 'Unnamed: 0'], axis=1, inplace=True)

    train_model(df)
