"""
build_db.py
Run this ONCE locally to pre-index all songs into database.pkl.
The Streamlit app will load this file instead of re-indexing
songs every time it starts — required since Q3B says the
database must "ship with the deployed app."
"""

import os
import glob
from fingerprint import build_database, save_database

# Point this to your actual songs folder
SONGS_DIR = "songs"

song_files = sorted(glob.glob(os.path.join(SONGS_DIR, "*.mp3")))
print(f"Found {len(song_files)} songs:")
for s in song_files:
    print(f"  - {os.path.basename(s)}")

db = build_database(song_files)
save_database(db, path="database.pkl")