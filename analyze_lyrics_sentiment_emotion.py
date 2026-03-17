import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# === CONFIG ===
INPUT_CSV = ""
OUTPUT_CSV = ""

# === INIT ANALYZERS ===
nltk.download('vader_lexicon')  # Download if not already present
sentiment_analyzer = SentimentIntensityAnalyzer()

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)
if "Sentiment" not in df.columns:
    df["Sentiment"] = ""
if "Dominant_Emotion" not in df.columns:
    df["Dominant_Emotion"] = ""

# === ANALYZE ===
for i, row in df.iterrows():
    lyrics = str(row["Lyrics"])
    if not lyrics.strip():
        continue

    # Sentiment
    sentiment_score = sentiment_analyzer.polarity_scores(lyrics)["compound"]
    sentiment = "positive" if sentiment_score >= 0.05 else (
        "negative" if sentiment_score <= -0.05 else "neutral"
    )
    df.at[i, "Sentiment"] = sentiment

    # Emotion (NRCLex)
    try:
        emotion_obj = NRCLex(lyrics)
        emotion_scores = emotion_obj.raw_emotion_scores
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "none"
    except Exception as e:
        print(f"[{i}] Emotion error for '{row['Title']}': {e}")
        dominant_emotion = "error"

    df.at[i, "Dominant_Emotion"] = dominant_emotion
    print(f"[{i}] {row['Title']} — Sentiment: {sentiment}, Emotion: {dominant_emotion}")

# === SAVE RESULTS ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Sentiment and emotion saved to {OUTPUT_CSV}")