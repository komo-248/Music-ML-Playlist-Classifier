import pandas as pd
import time
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import lyricsgenius

INPUT_CSV = ""
OUTPUT_CSV = ""

# === CONFIG ===
GENIUS_API_TOKEN = ""
LASTFM_API_KEY = ""
INPUT_LABELED_CSV = ""
INPUT_UNLABELED_CSV = ""
OUTPUT_LABELED_CSV = ""
OUTPUT_UNLABELED_CSV = ""
DELAY = 1 # seconds between API calls

# === NLTK SETUP ===
for pkg in ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']:
    nltk.download(pkg)
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# === Genius SETUP ===
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=15, retries=3)
genius.verbose = False

# === MISSING VALUE CHECK ===
def is_missing(val):
    return pd.isna(val) or str(val).strip().lower() in ["", "nan"]

# === LOAD CSVs AND PREPARE COLUMNS ===
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["Lyrics", "Tags", "Sentiment", "Dominant_Emotion"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    return df

df_labeled = load_and_prepare(INPUT_LABELED_CSV)
df_unlabeled = load_and_prepare(INPUT_UNLABELED_CSV)

# === LAST.FM API HELPERS ===
def get_lastfm_tags(title, artist):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    track_params = {
        "method": "track.gettoptags",
        "track": title,
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    try:
        response = requests.get(base_url, params=track_params)
        if response.status_code == 200:
            tags = response.json().get("toptags", {}).get("tag", [])
            return [t["name"] for t in tags[:3]] if isinstance(tags, list) else []
    except:
        pass
    return []

def get_artist_tags(artist):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    artist_params = {
        "method": "artist.gettoptags",
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    try:
        response = requests.get(base_url, params=artist_params)
        if response.status_code == 200:
            tags = response.json().get("toptags", {}).get("tag", [])
            return [t["name"] for t in tags[:3]] if isinstance(tags, list) else []
    except:
        pass
    return []

# === MAIN ENRICHMENT FUNCTION ===
def enrich_dataset(df, name=""):
    for i, row in df.iterrows():
        title, artist = str(row["Title"]), str(row["Artist"])
        print(f"\n[{name} {i}] Processing: {title} – {artist}")

        # === GENIUS LYRICS ===
        if is_missing(row["Lyrics"]):
            try:
                song = genius.search_song(title, artist)
                if song and song.lyrics:
                    df.at[i, "Lyrics"] = song.lyrics
                    print("  ✔ Lyrics added")
                else:
                    print("  ✘ Lyrics not found")
            except Exception as e:
                print(f"  ⚠ Genius error: {e}")
            time.sleep(DELAY)
        else:
            print("  ⏭️ Skipped Genius (lyrics already present)")

        # === LAST.FM TAGS ===
        if is_missing(row["Tags"]):
            tags = get_lastfm_tags(title, artist)
            if not tags:
                tags = get_artist_tags(artist)
            df.at[i, "Tags"] = ", ".join(tags)
            print(f"  ✔ Tags: {tags}")
            time.sleep(DELAY)
        else:
            print("  ⏭️ Skipped Tags (already present)")

        # === NLP SENTIMENT & EMOTION ===
        lyrics = row.get("Lyrics", "")
        if not is_missing(lyrics):
            needs_sentiment = is_missing(row.get("Sentiment", ""))
            needs_emotion = is_missing(row.get("Dominant_Emotion", ""))

            if needs_sentiment or needs_emotion:
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
                print("  ⏭️ Skipped NLP (already processed)")
        else:
            print("  ✘ Skipped NLP (no lyrics)")

    return df

# === ENRICH BOTH DATASETS ===
df_labeled = enrich_dataset(df_labeled, name="Labeled")
df_unlabeled = enrich_dataset(df_unlabeled, name="Unlabeled")

# === FINAL CLEANUP AND EXPORT ===
for df in [df_labeled, df_unlabeled]:
    df.replace("nan", "", inplace=True)
    df.fillna("", inplace=True)

df_labeled.to_csv(OUTPUT_LABELED_CSV, index=False)
df_unlabeled.to_csv(OUTPUT_UNLABELED_CSV, index=False)
print(f"\n✅ Done. Files saved to:\n - {OUTPUT_LABELED_CSV}\n - {OUTPUT_UNLABELED_CSV}")