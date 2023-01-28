import json
import math
import os
import subprocess
import pickle
import sys
from os.path import exists

import librosa
import pandas as pd
import requests
import configparser
from datetime import timedelta, datetime

from bson import Binary

from database import get_tracks_collection

from multiprocessing.dummy import Pool as ThreadPool

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)

config = configparser.ConfigParser()
config.read('config.ini')

APP_ID = config['tangy_credentials']['app_id']
API_KEY = config['tangy_credentials']['api_key']
BASE_URL = config['soundchart']['base_url']
NUM_DAYS = int(config['data_collection']['number_of_days'])

num_chroma_cols = 12
num_tonnetz_cols = 6
num_mel_cols = 128

chroma_columns = [f"Chroma_{i}" for i in range(1, num_chroma_cols+1)]
tonnetz_columns = [f"Tonnetz_{i}" for i in range(1, num_tonnetz_cols+1)]
mel_columns = [f"MEL_{i}" for i in range(1, num_mel_cols+1)]


def create_abt(tracks_data_set, low_level_features_df, feature_cols: list):
    tracks_data_set.drop(['_id', 'name', 'preview_url', 'release_date', 'uuid'], axis=1, inplace=True)
    tracks_data_set = tracks_data_set.rename(columns={'spotify_id': 'track_id'})

    feature_cols.append('track_id')
    low_level_features_df = low_level_features_df[feature_cols]

    merged = pd.merge(tracks_data_set, low_level_features_df, how='inner', on='track_id')
    return merged


def read_original_dataset():
    df = pd.read_csv('data/spotify_tracks.csv')
    return df[['id', 'name', 'popularity', 'speechiness', 'preview_url']]


def read_tracks_with_streams():
    df = pd.read_csv("data/tracks_with_streams.csv")
    return df


def download_all_samples(df, num_samples=None):
    if num_samples:
        for row in df.head(5).itertuples():
            download_sample(row.preview_url, row.name)
    else:
        for row in df.itertuples():
            download_sample(row.preview_url, row.name)


def download_sample(url, audio_name):
    ok = False
    try:
        audio_path = os.path.join('audio', audio_name)
        if url is not None and audio_path is not None:
            print(f"Downloading file: {audio_name}")
            request = f"curl {url} -o \"{audio_path}.mp3\""
            subprocess.call(request)
            ok = True
    except Exception as e:
        print(e)
    return ok


def prepare_headers():
    return {
        "x-app-id": APP_ID,
        "x-api-key": API_KEY
    }


def get_streams(uuid, from_date=None, to_date=datetime.now().date()):
    url = f"{BASE_URL}/api/v2/song/{uuid}/spotify/stream"
    params = {'startDate': from_date, 'endDate': to_date}
    result = requests.get(url, params=params, headers=prepare_headers())

    print(url)

    if not result.ok:
        print(f"Could not get streams for track with uuid {uuid} because: {json.loads(result.text)['errors'][0]['message']}")
        return None

    if result.json()['items']:
        return result.json()['items']

    return None


def get_track_from_soundchart(spotify_id) -> dict or None:
    url = f"{BASE_URL}/api/v2.8/song/by-platform/spotify/{spotify_id}"
    result = requests.get(url, headers=prepare_headers())

    print(url)

    if not result.ok:
        print(f"Could not resolve uuid for track with spotify id {spotify_id} because: {json.loads(result.text)}")
        return None

    return result.json()['object']


def remove_tracks_with_high_speechiness(df, threshold):
    return df.drop(df[df.speechiness > threshold].index)


def create_new_dataset_mongodb(df):
    pool = ThreadPool(5)
    tracks_collection = get_tracks_collection()

    def get_streams_for_track_n_last_days(track, n: int) -> int or None:
        assert (n <= 90)
        if not track['releaseDate']:
            return None

        n_days = timedelta(days=n)

        to_date = datetime.now()
        from_date = to_date - n_days  #datetime.fromisoformat(track['releaseDate']).date()

        # get total streams to today
        streams = get_streams(track['uuid'], from_date.date(), to_date.date())
        if streams:
            return abs(streams[0]['value'] - streams[-1]['value'])
        else:
            print(f"No streams found for track with uuid {track['uuid']}")
            return None

    def collect_uuid_streams_and_store_in_mongodb(row_tuple):
        _, row = row_tuple

        #check if track is already in db
        if tracks_collection.count_documents({'spotify_id': row['id']}):
            print(f"track with spotify id {row['id']} already in db")
            return

        track = get_track_from_soundchart(row['id'])

        if track:
            streams = get_streams_for_track_n_last_days(track, NUM_DAYS)
            if streams:
                tracks_collection.insert_one({
                    'spotify_id': row['id'],
                    'name': row['name'],
                    'preview_url': row['preview_url'],
                    'uuid': track['uuid'],
                    'release_date': track['releaseDate'],
                    'streams': streams
                })
            else:
                tracks_collection.insert_one({
                    'spotify_id': row['id'],  # store spotify id
                    'message': 'no streams'
                })
        else:
            tracks_collection.insert_one({
                'spotify_id': row['id'],   #store spotify id
                'message': 'no uuid'
            })

    results = pool.map(collect_uuid_streams_and_store_in_mongodb, df.iterrows())

    print(results)


def collect_data_set_with_streams():
    spotgen_df = read_original_dataset()
    spotgen_df = remove_tracks_with_high_speechiness(spotgen_df, threshold=0.66)
    create_new_dataset_mongodb(spotgen_df)


def calculate_output_shape(sr, hop_length, feature: str, sample_length_in_seconds=30):
    num_values = math.ceil((sr * sample_length_in_seconds) / hop_length)
    feature_column_map = dict(melspec=128, tonnetz=6)
    return feature_column_map[feature], num_values


def calculate_and_store_melspec(track):
    hop_length = 2048
    file_path = f"audio/{track.get('uuid')}.mp3"
    try:
        y, sr = librosa.load(f"audio/{track.get('uuid')}.mp3")

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           fmax=8000, hop_length=hop_length)

        if S.shape != calculate_output_shape(sr=sr, hop_length=hop_length, feature='melspec'):
            print("Melspec output was not of correct size. Skipping track...")
            print(calculate_output_shape(sr=sr, hop_length=hop_length, feature='melspec'))
            return

        pickled_array = Binary(pickle.dumps(S, protocol=2), subtype=128)

        add_field_to_track(track.get("_id"), "melspec", pickled_array)
    except Exception:
        print(f"Could not load file {track.get('name')} with path {file_path}")


def calculate_and_store_tonnetz(track):
    hop_length = 2048
    file_path = f"audio/{track.get('uuid')}.mp3"
    print(file_path)
    try:
        y, sr = librosa.load(file_path)

        S = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)

        if S.shape != calculate_output_shape(sr=sr, hop_length=hop_length, feature='tonnetz'):
            print("Tonnets output was not of correct size. Skipping track...")
            print(calculate_output_shape(sr=sr, hop_length=hop_length, feature='tonnetz'))
            return

        pickled_array = Binary(pickle.dumps(S, protocol=2), subtype=128)

        add_field_to_track(track.get("_id"), "tonnetz", pickled_array)
    except Exception as e:
        print(f"Could not load file {track.get('name')} with path {file_path} because {e}")


def add_field_to_track(track_id, field_name, value):
    get_tracks_collection().update_one({'_id': track_id}, {"$set": {field_name: value}})


def download_samples(num_samples=None):
    tracks_collection = get_tracks_collection().find({'melspec': {"$exists": False}})

    if num_samples:
        tracks_collection = tracks_collection.limit(num_samples)

    for t in tracks_collection:
        print(f"Track: {t.get('name'), t.get('streams'), t.get('preview_url')}")
        #print(pickle.loads(t.get('melspec')))

        if exists(f"audio/{t.get('uuid')}.mp3"):
            continue

        download_sample(t.get('preview_url'), t.get('uuid'))


def get_tracks_from_db(feature: str):
    return get_tracks_collection().find({"message": {"$ne": "no streams"},
                                         feature: {"$exists": True}},
                                        {

        "name": 0,
        "spotify_id": 0,
        "preview_url": 0,
        "uuid": 0,
        "message": 0,
        "release_date": 0
    })


if __name__ == "__main__":
    feature_columns = {
        'chromagram': chroma_columns,
        'tonnetz': tonnetz_columns,
        'mel_spectrogram': mel_columns,
    }

    #download_samples(num_samples=9000)

    tracks = get_tracks_collection()

    pool = ThreadPool(8)

    pool.map(calculate_and_store_melspec, tracks.find(
        {"message": {"$ne": "no streams"} })
             .limit(9000))

    # pool.map(calculate_and_store_tonnetz, tracks.find(
    #     {"message": {"$ne": "no streams"},
    #      'tonnetz': {"$exists": False}
    #      })
    #          .limit(9000))

    # y, sr = librosa.load(f"audio/0bc37e36-26e2-47f1-a600-06f48b94ea82.mp3")
    #
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=2048,
    #                                    fmax=8000)
    #
    # pickled_array = Binary(pickle.dumps(S, protocol=2), subtype=128)
    #
    # print(pickled_array)

    print("done!")



