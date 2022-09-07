from typing import Any, Mapping

import pymongo
from pymongo.database import Database


def connect_to_mongodb() -> Database[Mapping[str, Any] | Any]:
    client = pymongo.MongoClient("mongodb://localhost:27017")
    print(client.list_database_names())
    return client.get_database('song-stream-prediction')


def get_tracks_collection() -> pymongo.collection:
    db = connect_to_mongodb()
    tracks = db.get_collection('tracks')
    return tracks
