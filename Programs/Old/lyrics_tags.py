import pandas as pd
import lyricsgenius
import requests
import time

# === CONFIG ===
GENIUS_API_TOKEN = ""  # <-- Replace this
LASTFM_API_KEY = ""
INPUT_CSV = ""
OUTPUT_CSV = ""
DELAY = 1.5  # Avoid rate limiting

# === INIT GENIUS ===
genius = lyricsgenius.Genius(
    GENIUS_API_TOKEN,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    timeout=15,
    retries=3
)
genius.verbose = False

# === LOAD SONGS ===
df = pd.read_csv(INPUT_CSV)
if "Lyrics" not in df.columns:
    df["Lyrics"] = ""
if "Tags" not in df.columns:
    df["Tags"] = ""

# === LAST.FM TAG FETCHER ===
def get_tags_from_lastfm(title, artist):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    track_params = {
        "method": "track.gettoptags",
        "track": title,
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    response = requests.get(base_url, params=track_params)
    if response.status_code == 200:
        data = response.json()
        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, list):
            return [tag["name"] for tag in tags[:3]]
    return []

def get_artist_tags_from_lastfm(artist):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    artist_params = {
        "method": "artist.gettoptags",
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    response = requests.get(base_url, params=artist_params)
    if response.status_code == 200:
        data = response.json()
        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, list):
            return [tag["name"] for tag in tags[:3]]
    return []

# === FETCH LYRICS + TAGS ===
for i, row in df.iterrows():
    title, artist = str(row["Title"]), str(row["Artist"])
    lyrics_ok = bool(row["Lyrics"]) and pd.notna(row["Lyrics"])
    tags_ok = bool(row["Tags"]) and pd.notna(row["Tags"])

    # === Fetch Lyrics ===
    if not lyrics_ok:
        try:
            song = genius.search_song(title, artist)
            if song and song.lyrics:
                df.at[i, "Lyrics"] = song.lyrics
                print(f"[{i}] 🎤 Lyrics added: {title} - {artist}")
            else:
                print(f"[{i}] ❌ No lyrics: {title} - {artist}")
        except Exception as e:
            print(f"[{i}] ⚠️ Genius error: {title} - {artist}: {e}")
        time.sleep(DELAY)

    # === Fetch Tags ===
    if not tags_ok:
        tags = get_tags_from_lastfm(title, artist)
        if not tags:
            tags = get_artist_tags_from_lastfm(artist)
        df.at[i, "Tags"] = ", ".join(tags)
        print(f"[{i}] 🏷️ Tags added: {title} - {artist} → {tags}")
        time.sleep(DELAY)

# === SAVE OUTPUT ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Complete. Saved to {OUTPUT_CSV}")
