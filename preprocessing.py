import json
import os
import subprocess
import pandas as pd
import requests
import configparser
from datetime import timedelta, datetime

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


if __name__ == "__main__":
    feature_columns = {
        'chromagram': chroma_columns,
        'tonnetz': tonnetz_columns,
        'mel_spectrogram': mel_columns,
    }

    # for feature_name, feature_columns in feature_columns.items():
    #     features = pd.read_csv('data/low_level_audio_features.csv')
    #     tracks = pd.read_csv('data/tracks_with_streams.csv')
    #     abt = create_abt(tracks, features, feature_columns)
    #     print(abt.shape)
    #     print(abt.head())
    #     abt.to_csv(f"data/{feature_name}.csv")

    # features = pd.read_csv('data/low_level_audio_features.csv')
    # tracks = pd.read_csv('data/tracks_with_streams.csv')
    # abt = create_abt(tracks, features, chroma_columns + tonnetz_columns + mel_columns)
    # print(abt.shape)
    # print(abt.head())
    # abt.to_csv(f"data/mel_spec-tonnetz-chromagram.csv")

    # download_sample(
    #     "https://p.scdn.co/mp3-preview/95cb9df1b056d759920b5e85ad7f9aff0a390671?cid=b3cdb16d0df2409abf6a8f6c2f6c2e0c",
    #     "The Scientist")
