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
    return np.repeat(array_to_repeat, repeats=original_size).reshape(-1, original_size)


def calc_avg(feature):
    return np.array([apply_stat(feature[k], 'mean') for k in range(len(feature))])


def plot_feature(feature, y_axis, title, hop_length):
    fig, ax = plt.subplots()

    img = librosa.display.specshow(feature, y_axis=y_axis, x_axis='time', ax=ax, hop_length=hop_length)

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


def tonnetz(y, sr, hop_length=512):
    y = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
    print(tonnetz.shape)
    plot_feature(tonnetz, title='Tonal Centroids (Tonnetz)', y_axis='tonnetz', hop_length=hop_length)

    tonnetz_sum = calc_avg(tonnetz)
    tonnetz_sum = repeat_2d(tonnetz_sum, tonnetz.shape[1])

    print(tonnetz_sum.shape)

    plot_feature(tonnetz_sum, title='Tonal Centroids (Tonnetz) \"summarized\"', y_axis='tonnetz', hop_length=hop_length)


def plot_melspec(S, sr, title, hop_length):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', hop_length=hop_length,
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
    plt.show()


def melspec(y, sr, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length,
                                       fmax=8000)
    print(S.shape)

    plot_melspec(S, sr, title='Mel-frequency spectrogram', hop_length=hop_length)

    Ssum = calc_avg(S)
    Ssum = repeat_2d(Ssum, S.shape[1])

    print(Ssum.shape)

    plot_melspec(Ssum, sr, title='Mel-frequency spectrogram \"summarized\"', hop_length=hop_length)


def main():
    y, sr = librosa.load("audio/0bc37e36-26e2-47f1-a600-06f48b94ea82.mp3") #("audio/The Scientist.mp3")

    # fig, ax = plt.subplots()
    # librosa.display.waveshow(y, sr=sr, ax=ax)
    #
    # ax.set(title='Waveform')
    # plt.show()

    #chromagram(y, sr)
    tonnetz(y, sr, hop_length=2048)
    #melspec(y, sr, hop_length=4096)


if __name__ == "__main__":
    main()
