import sys

import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

from database import get_tracks_collection

np.set_printoptions(linewidth=500)
np.set_printoptions(threshold=sys.maxsize)


def apply_stat(value, stat='mean'):
    if stat == 'mean':
        res = np.mean(value)
    else:
        res = np.median(value)
    return res


def repeat_2d(array_to_repeat, original_size: int):
    return np.repeat(array_to_repeat, repeats=original_size).reshape(original_size, -1)


def calc_avg(feature):
    return np.array([apply_stat(feature[k], 'mean') for k in range(len(feature))])


def plot_feature(feature, y_axis, title):
    fig, ax = plt.subplots()

    img = librosa.display.specshow(feature, y_axis=y_axis, x_axis='time', ax=ax)

    ax.set(title=title)

    fig.colorbar(img, ax=ax)
    plt.show()


def chromagram(y, sr):
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    print(chroma_cq)
    plot_feature(chroma_cq, title='chroma_cqt', y_axis='chroma')

    print(type(chroma_cq))
    print(chroma_cq.shape)

    chroma_cq_sum = calc_avg(chroma_cq)
    chroma_cq_sum = repeat_2d(chroma_cq_sum, len(chroma_cq))

    #print(chroma_cq_sum)

    plot_feature(chroma_cq_sum, title='chroma_cqt \"summarized\"', y_axis='chroma')


def tonnetz(y, sr):
    y = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    print(tonnetz)
    plot_feature(tonnetz, title='Tonal Centroids (Tonnetz)', y_axis='tonnetz')

    tonnetz_sum = calc_avg(tonnetz)
    tonnetz_sum = repeat_2d(tonnetz_sum, len(tonnetz))

    plot_feature(tonnetz_sum, title='Tonal Centroids (Tonnetz) \"summarized\"', y_axis='tonnetz')


def plot_melspec(S, sr, title):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
    plt.show()


def melspec(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                       fmax=8000)
    print(S)

    plot_melspec(S, sr, title='Mel-frequency spectrogram')

    Ssum = calc_avg(S)
    Ssum = repeat_2d(Ssum, len(S))

    plot_melspec(Ssum, sr, title='Mel-frequency spectrogram \"summarized\"')


def main():
    y, sr = librosa.load("audio/The Scientist.mp3")

    # fig, ax = plt.subplots()
    # librosa.display.waveshow(y, sr=sr, ax=ax)
    #
    # ax.set(title='Waveform')
    # plt.show()

    #chromagram(y, sr)
    #tonnetz(y, sr)
    #melspec(y, sr)


if __name__ == "__main__":
    main()
