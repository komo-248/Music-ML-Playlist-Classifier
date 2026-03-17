"""
Music Playlist Classifier
=========================
Multi-label song classifier using Sentence-BERT embeddings and a
Multi-output MLP neural network. Predicts which playlists a song belongs
to based on lyrics, genre tags, sentiment, and emotion features.

Run in Google Colab with a GPU runtime for best performance.

Required files (upload to Google Drive):
  - labeled_dataset.csv   (songs already assigned to playlists)
  - unlabeled_dataset.csv (full song library, no labels)

Output:
  - predicted_playlists.csv
"""

# =============================================================================
# 1. INSTALL & IMPORT
# =============================================================================

# Run this cell first in Colab
# !pip install sentence-transformers scikit-learn lightgbm

import pandas as pd
import numpy as np
import os
from ast import literal_eval

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample

# Mount Google Drive (Colab only)
from google.colab import drive
drive.mount('/content/drive')


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Update these paths to match your Google Drive layout
LABELED_CSV   = '/content/drive/MyDrive/labeled_dataset.csv'
UNLABELED_CSV = '/content/drive/MyDrive/unlabeled_dataset.csv'
OUTPUT_DIR    = '/content/drive/MyDrive/predicted_playlists_final'
OUTPUT_CSV    = os.path.join(OUTPUT_DIR, 'predicted_playlists.csv')

# Playlists the model will predict
VALID_PLAYLISTS = [
    'rap_playlist',
    'country_playlist',
    'pop_playlist',
    'lofi_playlist',
    'feels_playlist',
    'edm_alt_rock_playlist',
    'christmas_playlist',
    'movie_playlist',
    'star_playlist',
    'christian_playlist',
    'vocalist_instrumental_playlist',
]

# Per-label classification thresholds (0.0 - 1.0)
# Lower = more inclusive, Higher = more precise
# Broad genres use 0.5 to avoid over-assignment
THRESHOLDS = {label: 0.15 for label in VALID_PLAYLISTS}
THRESHOLDS.update({
    'rap_playlist':     0.5,
    'country_playlist': 0.5,
})


# =============================================================================
# 3. LOAD DATA
# =============================================================================

labeled   = pd.read_csv(LABELED_CSV)
unlabeled = pd.read_csv(UNLABELED_CSV)

print(f"Labeled songs:   {len(labeled):,}")
print(f"Unlabeled songs: {len(unlabeled):,}")


# =============================================================================
# 4. PARSE PLAYLIST LABELS
# =============================================================================
# The Playlists column may be stored as a Python list string e.g. "['rap', 'pop']"
# or as a comma-separated string. This handles both.

def parse_playlists(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            val = literal_eval(x)
            if isinstance(val, list):
                return val
        except Exception:
            pass
        return [item.strip() for item in x.split(',') if item.strip()]
    return []

labeled['Playlists'] = labeled['Playlists'].apply(parse_playlists)
labeled['Add to new playlists?'] = labeled['Add to new playlists?'].fillna(False)


# =============================================================================
# 5. COMBINE TEXT FOR EMBEDDING
# =============================================================================
# Each song is represented as a single string combining all text features.
# Sentence-BERT encodes this into a dense 384-dim semantic vector.

def combine_text(row):
    return ' '.join([
        str(row.get('Lyrics', '')),
        str(row.get('Tags', '')),
        str(row.get('Sentiment', '')),
        str(row.get('Dominant_Emotion', '')),
    ])

labeled['Combined']   = labeled.apply(combine_text, axis=1)
unlabeled['Combined'] = unlabeled.apply(combine_text, axis=1)

# Full dataset = labeled + unlabeled (predictions run over everything)
full_df = pd.concat([labeled, unlabeled], ignore_index=True)
full_df['Combined'] = full_df.apply(combine_text, axis=1)


# =============================================================================
# 6. GENERATE SENTENCE-BERT EMBEDDINGS
# =============================================================================
# all-MiniLM-L6-v2: fast, lightweight, strong semantic similarity performance.
# Encodes each combined text string → 384-dimensional float vector.

print("Loading Sentence-BERT model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding labeled songs...")
X_train = embedder.encode(
    labeled['Combined'].tolist(),
    batch_size=64,
    show_progress_bar=True
)

print("Encoding full dataset...")
X_full = embedder.encode(
    full_df['Combined'].tolist(),
    batch_size=64,
    show_progress_bar=True
)

print(f"Embedding shape (train): {X_train.shape}")
print(f"Embedding shape (full):  {X_full.shape}")


# =============================================================================
# 7. BUILD TARGET LABELS
# =============================================================================

mlb = MultiLabelBinarizer(classes=VALID_PLAYLISTS)
Y   = mlb.fit_transform(labeled['Playlists'])

print(f"\nPlaylist label distribution (training set):")
label_counts = pd.DataFrame(Y, columns=mlb.classes_).sum().sort_values(ascending=False)
print(label_counts.to_string())


# =============================================================================
# 8. CLASS BALANCING (UPSAMPLING)
# =============================================================================
# Playlist sizes are highly imbalanced (lofi >> christmas).
# Each class is upsampled with replacement to match the largest class,
# preventing the model from ignoring minority playlists.

X_df       = pd.DataFrame(X_train)
Y_df       = pd.DataFrame(Y, columns=mlb.classes_)
combined   = pd.concat([X_df, Y_df], axis=1)
max_size   = int(Y_df.sum().max())

print(f"\nUpsampling all classes to {max_size} samples...")
balanced_rows = []
for label in mlb.classes_:
    subset    = combined[combined[label] == 1]
    upsampled = resample(subset, replace=True, n_samples=max_size, random_state=42)
    balanced_rows.append(upsampled)

balanced   = pd.concat(balanced_rows, ignore_index=True)
X_balanced = balanced.iloc[:, :X_train.shape[1]].values
Y_balanced = balanced[list(mlb.classes_)].values

print(f"Balanced training set size: {len(X_balanced):,} rows")


# =============================================================================
# 9. TRAIN CLASSIFIER
# =============================================================================
# MultiOutputClassifier wraps one binary MLP per playlist label.
# Architecture: 384 → 256 → 128 → 1 (per label)

print("\nTraining MLP classifier...")
mlp   = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
model = MultiOutputClassifier(mlp)
model.fit(X_balanced, Y_balanced)
print("Training complete.")


# =============================================================================
# 10. PREDICT
# =============================================================================
# Each label's classifier returns a probability. Predictions are made by
# comparing against per-label thresholds rather than a fixed 0.5 cutoff.

print("\nRunning predictions...")
probas      = model.predict_proba(X_full)
num_samples = len(X_full)
num_labels  = len(probas)
preds       = np.zeros((num_samples, num_labels), dtype=int)

for i, label_probs in enumerate(probas):
    label   = mlb.classes_[i]
    thresh  = THRESHOLDS.get(label, 0.15)

    # predict_proba returns shape (n_samples, 2) — column 1 is P(positive)
    if isinstance(label_probs, np.ndarray) and label_probs.ndim == 2:
        positive_probs = label_probs[:, 1]
    else:
        positive_probs = np.array(label_probs)

    preds[:, i] = (positive_probs >= thresh).astype(int)

raw_preds = mlb.inverse_transform(preds)


# =============================================================================
# 11. CLEAN & EXPORT PREDICTIONS
# =============================================================================
# Parse any malformed list strings in predictions, then filter:
# - Labeled songs: only include if "Add to new playlists?" is True
# - Unlabeled songs: always include
# Songs with no predicted playlist are tagged "unassigned"

def clean_pred(entry):
    """Normalize prediction tuples — handles edge case of stringified lists."""
    if (isinstance(entry, tuple) and len(entry) == 1
            and isinstance(entry[0], str) and entry[0].startswith('[')):
        try:
            return tuple(literal_eval(entry[0]))
        except Exception:
            pass
    return entry

cleaned_preds = [clean_pred(p) for p in raw_preds]

os.makedirs(OUTPUT_DIR, exist_ok=True)
rows = []

for idx, playlists in enumerate(cleaned_preds):
    title     = full_df.loc[idx, 'Title']
    artist    = full_df.loc[idx, 'Artist']
    is_labeled = idx < len(labeled)
    include   = (not is_labeled) or (labeled.loc[idx, 'Add to new playlists?'] == True)

    if not include:
        continue

    if playlists:
        for p in playlists:
            if p in VALID_PLAYLISTS:
                rows.append({'Title': title, 'Artist': artist, 'Playlist': p})
    else:
        rows.append({'Title': title, 'Artist': artist, 'Playlist': 'unassigned'})

output_df = pd.DataFrame(rows)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Done. {len(output_df):,} rows saved to: {OUTPUT_CSV}")
print(output_df['Playlist'].value_counts().to_string())
