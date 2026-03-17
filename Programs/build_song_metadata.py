import pandas as pd
import time
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import lyricsgenius

# === CONFIG ===
GENIUS_API_TOKEN = ""
LASTFM_API_KEY = ""
INPUT_CSV = ""
OUTPUT_CSV = ""
DELAY = 0.1 # seconds between API calls

# === NLTK DOWNLOADS (Ensure Required Data) ===
for pkg in ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']:
    nltk.download(pkg)
nltk.download('vader_lexicon')

# === INITIALIZE ANALYZERS ===
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=15, retries=3)
genius.verbose = False
vader = SentimentIntensityAnalyzer()

# === LOAD BASE SONG LIST ===
df = pd.read_csv(INPUT_CSV)
df["Lyrics"] = ""
df["Tags"] = ""
df["Sentiment"] = ""
df["Dominant_Emotion"] = ""

# === HELPER: Last.fm Track Tags ===
def get_lastfm_tags(title, artist):
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
        tags = response.json().get("toptags", {}).get("tag", [])
        return [t["name"] for t in tags[:3]] if isinstance(tags, list) else []
    return []

# === HELPER: Fallback to Artist Tags ===
def get_artist_tags(artist):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    artist_params = {
        "method": "artist.gettoptags",
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    response = requests.get(base_url, params=artist_params)
    if response.status_code == 200:
        tags = response.json().get("toptags", {}).get("tag", [])
        return [t["name"] for t in tags[:3]] if isinstance(tags, list) else []
    return []

# === PROCESS SONGS ===
for i, row in df.iterrows():
    title, artist = str(row["Title"]), str(row["Artist"])
    print(f"\n[{i}] Processing: {title} – {artist}")

    # === GENIUS LYRICS ===
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            df.at[i, "Lyrics"] = song.lyrics
            print(f"  ✔ Lyrics added")
        else:
            print(f"  ✘ Lyrics not found")
    except Exception as e:
        print(f"  ⚠ Genius error: {e}")
    time.sleep(DELAY)

    # === LAST.FM TAGS ===
    tags = get_lastfm_tags(title, artist)
    if not tags:
        tags = get_artist_tags(artist)
    df.at[i, "Tags"] = ", ".join(tags)
    print(f"  ✔ Tags: {tags}")
    time.sleep(DELAY)

    # === NLP: SENTIMENT + EMOTION ===
    lyrics = df.at[i, "Lyrics"]
    if lyrics:
        sentiment_score = vader.polarity_scores(lyrics)["compound"]
        sentiment = "positive" if sentiment_score >= 0.05 else ("negative" if sentiment_score <= -0.05 else "neutral")
        df.at[i, "Sentiment"] = sentiment

        try:
            emotion_obj = NRCLex(lyrics)
            emotion_scores = emotion_obj.raw_emotion_scores
            dominant = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "none"
        except:
            dominant = "error"

        df.at[i, "Dominant_Emotion"] = dominant
        print(f"  ✔ Sentiment: {sentiment}, Emotion: {dominant}")
    else:
        print(f"  ✘ Skipped NLP (no lyrics)")

# === SAVE RESULTS ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done. Enriched dataset saved to: {OUTPUT_CSV}")
