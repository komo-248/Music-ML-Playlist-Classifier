# Music ML – Genre & Playlist Classifier

A multi-label song classifier that predicts which personal playlists a song belongs to, built using sentence embeddings, NLP sentiment/emotion analysis, and a multi-output MLP neural network. The project also includes a full data recovery pipeline using OCR and LLM-assisted cleaning to reconstruct a corrupted music library database from screenshots.

---

## Table of Contents

- [Overview](#overview)
- [The Problem: Corrupted Database](#the-problem-corrupted-database)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1 – OCR Extraction](#stage-1--ocr-extraction)
- [Stage 2 – OCR Cleaning & MusicBrainz Validation](#stage-2--ocr-cleaning--musicbrainz-validation)
- [Stage 3 – Metadata Enrichment](#stage-3--metadata-enrichment)
- [Stage 4 – Classification (Colab)](#stage-4--classification-colab)
- [Playlists](#playlists)
- [Results](#results)
- [Dependencies](#dependencies)
- [Files](#files)

---

## Overview

The end goal was to take a library of ~18,000 songs and automatically assign each one to one or more personal genre/mood playlists. The training signal came from existing hand-curated playlists; the model learned the textual and emotional signature of each playlist and applied it to the full unclassified library.

**Techniques used:**
- OCR with Tesseract for database recovery from screenshots
- MusicBrainz API for fuzzy song title/artist validation
- Genius API for lyric retrieval
- Last.fm API for genre tag retrieval
- VADER sentiment analysis (NLTK)
- NRCLex emotion classification
- Sentence-BERT embeddings (`all-MiniLM-L6-v2`)
- Multi-output MLP classifier (scikit-learn) with class balancing

---

## The Problem: Corrupted Database

The song library was originally managed in a YouTube Music frontend that supported database export. When migrating to Apple Music, the exported database file was corrupted and unrecoverable. Rather than re-entering thousands of songs manually, the recovery process was:

1. **Screenshot** every page of the music library from the frontend UI
2. **OCR** all screenshots with Tesseract to extract raw text
3. **Clean** the OCR output using regex heuristics and MusicBrainz API validation to reconstruct proper `Title – Artist` pairs
4. Use the recovered song list as the base dataset for enrichment and classification

---

## Pipeline Overview

```
Screenshots (PNG)
      │
      ▼
[Stage 1] OCR Extraction (Tesseract)
      │  → OCR.txt
      ▼
[Stage 2] OCR Cleaning + MusicBrainz Validation
      │  → Cleaned_Song_List.csv
      ▼
[Stage 3] Metadata Enrichment
      │    Genius API → Lyrics
      │    Last.fm API → Genre Tags
      │    VADER → Sentiment
      │    NRCLex → Dominant Emotion
      │  → Full_Song_Metadata.csv
      │  → labeled_dataset.csv (songs in existing playlists)
      │  → unlabeled_dataset.csv (full library, no labels)
      ▼
[Stage 4] Classification (Google Colab)
      │    Sentence-BERT embeddings
      │    Multi-output MLP + class balancing
      │    Per-label probability thresholds
      │  → predicted_playlists.csv
```

---

## Stage 1 – OCR Extraction

**Script:** `OCR/Step 1 (Extractor)/ocr_song_extractor.py`

Runs Tesseract OCR over a folder of `.png` screenshots and writes all extracted text to a single output file, with each image's output delimited by a header block.

```python
# === USER CONFIGURATION ===
INPUT_FOLDER = r"C:\path\to\screenshots"   # Folder of .png screenshot files
OUTPUT_FILE  = r"C:\path\to\OCR.txt"       # Output raw text file
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Output format (`OCR.txt`):**
```
--- screenshot_001.png ---
Song Title
Artist Name
3:42
...

--- screenshot_002.png ---
...
```

**Requirements:**
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed
- `pytesseract`, `Pillow`

---

## Stage 2 – OCR Cleaning & MusicBrainz Validation

**Script:** `OCR/Step 2 (Clean)/clean_ocr_music_blocks.py`

The raw OCR output is noisy — timestamps, UI text, and partial reads all contaminate the song data. This script cleans the output per image block using regex heuristics, then optionally validates each entry against the MusicBrainz database to confirm and correct artist/title pairs.

**Cleaning logic:**
- Strips lines under 2 characters
- Removes timestamp patterns (e.g. `3:42`)
- Pairs adjacent short lines as `Title – Artist` candidates
- Fuzzy-searches MusicBrainz for unmatched entries and returns the top result

```python
USE_MUSICBRAINZ = True  # Set False to skip API validation (faster, less accurate)
```

> MusicBrainz requests are throttled to 1 per second to comply with API rate limits.

**Output:** `Cleaned_Songs_By_Image.txt` → reviewed and exported as `Cleaned_Song_List.csv`

---

## Stage 3 – Metadata Enrichment

**Scripts:** `Programs/build_song_metadata.py`, `Programs/build_song_metadata_after currentplaylists.py`

Each song in the cleaned list is enriched with four additional features used as classifier input:

| Feature | Source | Method |
|---------|--------|--------|
| `Lyrics` | Genius API | `lyricsgenius` — searched by title + artist |
| `Tags` | Last.fm API | `track.gettoptags` → fallback to `artist.gettoptags` |
| `Sentiment` | NLTK VADER | Compound score → `positive` / `neutral` / `negative` |
| `Dominant_Emotion` | NRCLex | Top emotion from NRC Emotion Lexicon scores |

**Configuration (set before running):**

```python
GENIUS_API_TOKEN = ""   # Genius API token
LASTFM_API_KEY   = ""   # Last.fm API key
INPUT_CSV        = ""   # Path to cleaned song list
OUTPUT_CSV       = ""   # Path for enriched output
DELAY = 1               # Seconds between API calls (avoid rate limiting)
```

The second script (`build_song_metadata_after currentplaylists.py`) handles the labeled and unlabeled datasets separately and skips already-enriched rows, making it safe to resume interrupted runs.

**Output columns:**

```
Title | Artist | Lyrics | Tags | Sentiment | Dominant_Emotion | Playlists
```

- `labeled_dataset.csv` — songs already assigned to at least one playlist (~58,997 rows)
- `unlabeled_dataset.csv` — full library with no playlist assignment (~182,439 rows)

---

## Stage 4 – Classification (Colab)

**Notebook:** `ML/Full Classifier/Untitled0.ipynb`

Run in Google Colab with GPU. Takes the labeled and unlabeled datasets and trains a multi-label MLP classifier to predict playlist membership for every song.

### Embedding

Each song's text fields (lyrics, tags, sentiment, emotion) are concatenated and encoded into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`:

```python
embedder = SentenceTransformer('all-MiniLM-L6-v2')
combined_text = ' '.join([lyrics, tags, sentiment, dominant_emotion])
X = embedder.encode(combined_texts, batch_size=64)
```

Sentence-BERT was chosen over TF-IDF or word2vec because it produces semantically meaningful dense representations — words like "heartbreak" and "longing" end up near each other in embedding space, which helps the model generalize across lyric style.

### Class Balancing

Playlist sizes vary widely (e.g. `lofi_playlist` >> `christmas_playlist`). Without balancing, the model learns to ignore minority classes. Each playlist class is upsampled to match the largest class using `sklearn.utils.resample` with replacement:

```python
max_size = Y_df.sum().max()
for label in mlb.classes_:
    rows = combined_df[combined_df[label] == 1]
    upsampled = resample(rows, replace=True, n_samples=max_size, random_state=42)
```

### Model

```python
MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
wrapped in MultiOutputClassifier  # One binary classifier per playlist
```

### Prediction Thresholds

Rather than a fixed 0.5 threshold, per-label thresholds were tuned to control precision/recall trade-offs per playlist:

```python
thresholds = {label: 0.15 for label in mlb.classes_}
thresholds.update({
    'rap_playlist':     0.5,   # High precision needed — broad genre bleeds
    'country_playlist': 0.5,   # Same reason
})
```

Lower thresholds (0.15) for mood/niche playlists allow more inclusive assignment; higher thresholds (0.5) for broad genres prevent over-assignment.

---

## Playlists

The classifier predicts membership across 11 playlists:

| Playlist | Type |
|----------|------|
| `rap_playlist` | Genre |
| `country_playlist` | Genre |
| `pop_playlist` | Genre |
| `edm_alt_rock_playlist` | Genre |
| `christian_playlist` | Genre/Theme |
| `christmas_playlist` | Theme/Seasonal |
| `lofi_playlist` | Mood/Style |
| `feels_playlist` | Mood |
| `movie_playlist` | Context/Theme |
| `star_playlist` | Curated favorites |
| `vocalist_instrumental_playlist` | Style |

---

## Results

**Output:** `Final Predicted Playlists/predicted_playlists.csv`

The final CSV contains each song's predicted playlist assignments. Songs flagged `Add to new playlists? = True` in the labeled set, and all unlabeled songs, are included in the output. Songs with no playlist prediction above threshold are tagged `unassigned`.

| Title | Artist | Playlist |
|-------|--------|---------|
| 'Til You Can't | Cody Johnson | country_playlist |
| ... | ... | ... |

---

## Dependencies

```bash
pip install pytesseract Pillow
pip install musicbrainzngs tqdm
pip install lyricsgenius requests
pip install nltk nrclex pandas
pip install sentence-transformers scikit-learn
```

**External tools:**
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Stage 1)
- Genius API token — [genius.com/api-clients](https://genius.com/api-clients)
- Last.fm API key — [last.fm/api](https://www.last.fm/api)

---

## Files

```
├── OCR/
│   ├── Step 1 (Extractor)/
│   │   └── ocr_song_extractor.py          # Tesseract OCR over screenshots
│   ├── Step 2 (Clean)/
│   │   └── clean_ocr_music_blocks.py      # Regex cleaning + MusicBrainz validation
│   ├── Song Titles.txt                    # Raw recovered song list
│   └── Cleaned_Song_List.csv              # Cleaned Title/Artist pairs
│
├── Programs/
│   ├── build_song_metadata.py             # Initial metadata enrichment
│   ├── build_song_metadata_after currentplaylists.py  # Resume-safe enrichment
│   └── Old/
│       ├── analyze_lyrics_sentiment_emotion.py  # Standalone sentiment/emotion script
│       └── lyrics_tags.py                       # Standalone lyrics + tags script
│
├── ML/
│   ├── Full_Song_Metadata.csv             # Full enriched dataset
│   ├── labeled_dataset.csv                # Songs with playlist labels (~59k rows)
│   ├── unlabeled_dataset.csv              # Full library without labels (~182k rows)
│   └── Full Classifier/
│       └── Untitled0.ipynb                # Colab training + prediction notebook
│
├── Playlists/
│   ├── Current/                           # Full playlist CSVs used for training
│   └── Playlists First 100/               # Initial 100-song playlist text files
│
└── Final Predicted Playlists/
    └── predicted_playlists.csv            # Final classifier output
```
