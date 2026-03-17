# Music ML – Playlist Classifier

Multi-label song classifier that predicts which playlists a song belongs to using Sentence-BERT embeddings, NLP-derived features, and a multi-output MLP neural network. Built to classify a personal library of ~18,000 songs across 11 genre and mood playlists.

Also includes a full data recovery pipeline using OCR and MusicBrainz validation — used to reconstruct a music library database that was corrupted during a platform migration.

---

## Table of Contents

- [Project Background](#project-background)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Pipeline](#pipeline)
  - [Step 1 — OCR Extraction](#step-1--ocr-extraction)
  - [Step 2 — OCR Cleaning](#step-2--ocr-cleaning)
  - [Step 3 — Metadata Enrichment](#step-3--metadata-enrichment)
  - [Step 4 — Classification](#step-4--classification)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Playlists](#playlists)
- [Output](#output)

---

## Project Background

The song library was originally managed in a YouTube Music frontend that supported database export. When migrating to Apple Music, the exported database file was corrupted and unrecoverable. Rather than re-entering thousands of songs manually, the recovery process was:

1. Screenshot every page of the music library from the UI
2. OCR all screenshots with Tesseract to extract raw text
3. Clean the OCR output and validate against MusicBrainz to reconstruct `Title – Artist` pairs
4. Enrich the recovered song list with lyrics, genre tags, sentiment, and emotion
5. Train a multi-label classifier on hand-curated playlists and predict assignments for the full library

---

## Repo Structure

```
Music-ML-Playlist-Classifier/
│
├── OCR/
│   ├── Step 1 (Extractor)/
│   │   └── ocr_song_extractor.py          # Tesseract OCR over screenshots → OCR.txt
│   └── Step 2 (Clean)/
│       └── clean_ocr_music_blocks.py      # Regex cleaning + MusicBrainz validation
│
├── Programs/
│   ├── build_song_metadata.py             # Enrich song list with lyrics, tags, NLP
│   ├── build_song_metadata_resume.py      # Resume-safe version (skips already enriched rows)
│   └── Old/
│       ├── lyrics_tags.py                 # Standalone lyrics + Last.fm tags script
│       └── analyze_lyrics_sentiment_emotion.py  # Standalone NLP sentiment/emotion script
│
├── ML/
│   └── playlist_classifier.py            # Full training + prediction script (run in Colab)
│
├── data/
│   ├── labeled_dataset.csv               # Songs with playlist labels (~59k rows)
│   ├── unlabeled_dataset.csv             # Full library, no labels (~182k rows)
│   └── predicted_playlists.csv           # Final classifier output
│
└── requirements.txt
```

> `data/` files are not included in the repo due to size. See [Datasets](#datasets) for column schema.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/komo-248/Music-ML-Playlist-Classifier.git
cd Music-ML-Playlist-Classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. External tools (OCR steps only)**

Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and update the path in `ocr_song_extractor.py`:
```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**4. API keys (metadata enrichment only)**

The enrichment scripts require two API keys — set them directly in the config section of each script before running:

| Key | Where to get it |
|-----|----------------|
| `GENIUS_API_TOKEN` | [genius.com/api-clients](https://genius.com/api-clients) |
| `LASTFM_API_KEY` | [last.fm/api](https://www.last.fm/api) |

> Never commit API keys to the repo. Set them locally in the script config before running.

---

## Pipeline

The four steps below describe the full data flow from screenshots to classified playlists. Each step can be run independently if you already have the outputs from a previous step.

---

### Step 1 — OCR Extraction

**Script:** `OCR/Step 1 (Extractor)/ocr_song_extractor.py`  
**Input:** Folder of `.png` screenshots of the music library UI  
**Output:** `OCR.txt` — raw text extracted per screenshot

Runs Tesseract over every `.png` in the input folder and writes the results to a single output file. Each screenshot's output is separated by a filename header.

**Configure before running:**
```python
INPUT_FOLDER   = r"C:\path\to\screenshots"
OUTPUT_FILE    = r"C:\path\to\OCR.txt"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Run:**
```bash
python "OCR/Step 1 (Extractor)/ocr_song_extractor.py"
```

**Output format:**
```
--- screenshot_001.png ---
Song Title
Artist Name
3:42
...
```

---

### Step 2 — OCR Cleaning

**Script:** `OCR/Step 2 (Clean)/clean_ocr_music_blocks.py`  
**Input:** `OCR.txt` from Step 1  
**Output:** `Cleaned_Songs_By_Image.txt` → reviewed and saved as `Cleaned_Song_List.csv`

Raw OCR output contains timestamps, UI text, and partial reads. This script cleans it per image block and optionally validates each entry against MusicBrainz to confirm and correct `Title – Artist` pairs.

**Configure before running:**
```python
OCR_INPUT_PATH      = r"C:\path\to\OCR.txt"
OUTPUT_CLEANED_PATH = r"C:\path\to\Cleaned_Songs_By_Image.txt"
USE_MUSICBRAINZ     = True  # Set False to skip validation (faster, less accurate)
```

**Cleaning logic:**
- Removes lines under 2 characters and timestamp patterns (e.g. `3:42`)
- Pairs adjacent short lines as `Title – Artist` candidates
- For unmatched entries, fuzzy-searches MusicBrainz and returns the top result
- Requests are throttled to 1/sec to comply with MusicBrainz rate limits

**Run:**
```bash
python "OCR/Step 2 (Clean)/clean_ocr_music_blocks.py"
```

---

### Step 3 — Metadata Enrichment

**Script:** `Programs/build_song_metadata_resume.py`  
*(Use `build_song_metadata.py` for a fresh run from scratch)*  
**Input:** `Cleaned_Song_List.csv` with `Title` and `Artist` columns  
**Output:** `labeled_dataset.csv`, `unlabeled_dataset.csv` — enriched with 4 features

Each song is enriched with the features that the classifier uses as input:

| Column | Source | Description |
|--------|--------|-------------|
| `Lyrics` | Genius API | Full lyrics fetched via `lyricsgenius` |
| `Tags` | Last.fm API | Top 3 genre/style tags for track, fallback to artist |
| `Sentiment` | NLTK VADER | `positive` / `neutral` / `negative` from compound score |
| `Dominant_Emotion` | NRCLex | Top emotion from NRC Lexicon (joy, fear, anger, etc.) |

**Configure before running:**
```python
GENIUS_API_TOKEN     = ""  # Your Genius API token
LASTFM_API_KEY       = ""  # Your Last.fm API key
INPUT_LABELED_CSV    = ""  # Path to labeled songs CSV
INPUT_UNLABELED_CSV  = ""  # Path to full library CSV
OUTPUT_LABELED_CSV   = ""  # Output path for enriched labeled set
OUTPUT_UNLABELED_CSV = ""  # Output path for enriched unlabeled set
DELAY = 1                  # Seconds between API calls (avoid rate limiting)
```

**Run:**
```bash
python Programs/build_song_metadata_resume.py
```

The resume-safe version skips rows that already have lyrics, tags, or NLP scores — safe to restart after interruptions without re-fetching or wasting API calls.

---

### Step 4 — Classification

**Script:** `ML/playlist_classifier.py`  
**Recommended:** Google Colab with GPU runtime  
**Input:** `labeled_dataset.csv`, `unlabeled_dataset.csv`  
**Output:** `predicted_playlists.csv`

#### Running in Google Colab

1. Upload `labeled_dataset.csv` and `unlabeled_dataset.csv` to Google Drive
2. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook
3. Set runtime to **GPU**: Runtime → Change runtime type → T4 GPU
4. In the first cell, install dependencies:
   ```python
   !pip install sentence-transformers scikit-learn lightgbm
   ```
5. Copy the contents of `playlist_classifier.py` into the next cell
6. Update the file paths in the Configuration section:
   ```python
   LABELED_CSV   = '/content/drive/MyDrive/labeled_dataset.csv'
   UNLABELED_CSV = '/content/drive/MyDrive/unlabeled_dataset.csv'
   OUTPUT_DIR    = '/content/drive/MyDrive/predicted_playlists_final'
   ```
7. Run all cells — the output CSV will be saved to your Drive

#### How it works

**Embedding:** Lyrics, tags, sentiment, and emotion are concatenated into a single string per song and encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`. Sentence-BERT captures semantic meaning so songs with similar lyrical themes cluster together in vector space, even when the exact words differ.

**Class balancing:** Playlist sizes vary widely — `lofi_playlist` is much larger than `christmas_playlist`. Without correction the model ignores minority classes. Each class is upsampled with replacement to match the largest class before training.

**Model:** `MLPClassifier(hidden_layer_sizes=(256, 128))` wrapped in `MultiOutputClassifier` — one independent binary classifier per playlist.

**Thresholds:** Each playlist uses its own probability threshold rather than a fixed 0.5:

| Playlist | Threshold | Reason |
|----------|-----------|--------|
| Most playlists | 0.15 | Inclusive — mood and niche playlists benefit from a wider net |
| `rap_playlist` | 0.50 | Stricter — broad genre bleeds into many other categories |
| `country_playlist` | 0.50 | Same reason |

---

## Datasets

Data files are not included in the repo due to size. Column schema:

**`labeled_dataset.csv`** (~59,000 rows)
```
Title | Artist | Lyrics | Tags | Sentiment | Dominant_Emotion | Playlists | Add to new playlists?
```

**`unlabeled_dataset.csv`** (~182,000 rows)
```
Title | Artist | Lyrics | Tags | Sentiment | Dominant_Emotion | Playlists
```

**`predicted_playlists.csv`** (output)
```
Title | Artist | Playlist
```

---

## Model Details

| Component | Details |
|-----------|---------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Classifier | `MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300)` |
| Wrapper | `MultiOutputClassifier` — one binary classifier per label |
| Balancing | Upsample with replacement to max class size |
| Runtime | ~10 min on Colab T4 GPU for 240k songs |

---

## Playlists

| Playlist | Type |
|----------|------|
| `rap_playlist` | Genre |
| `country_playlist` | Genre |
| `pop_playlist` | Genre |
| `edm_alt_rock_playlist` | Genre |
| `christian_playlist` | Theme |
| `christmas_playlist` | Seasonal |
| `lofi_playlist` | Mood/Style |
| `feels_playlist` | Mood |
| `movie_playlist` | Context |
| `star_playlist` | Curated favorites |
| `vocalist_instrumental_playlist` | Style |

---

## Output

`predicted_playlists.csv` contains one row per song-playlist assignment. Songs with no prediction above threshold are tagged `unassigned`.

```
Title,Artist,Playlist
'Til You Can't,Cody Johnson,country_playlist
...
```

Songs in the labeled set are only included in output if `Add to new playlists?` is `True`. All unlabeled songs are always included.
