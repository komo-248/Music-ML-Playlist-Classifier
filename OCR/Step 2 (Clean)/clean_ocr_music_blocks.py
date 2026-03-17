import re
import time
from collections import defaultdict
from tqdm import tqdm
import musicbrainzngs

# === CONFIGURATION ===
OCR_INPUT_PATH = r"C:\path\to\OCR.txt"
OUTPUT_CLEANED_PATH = r"C:\path\to\Cleaned_Songs_By_Image.txt"
USE_MUSICBRAINZ = True  # Set to False to disable artist/title lookup
# ======================

# Setup MusicBrainz API
musicbrainzngs.set_useragent("OCRMusicCleaner", "1.0", "your-email@example.com")

def load_ocr_blocks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r"--- ([^\n]+) ---", content)
    grouped = [(blocks[i].strip(), blocks[i+1].strip()) for i in range(1, len(blocks)-1, 2)]
    return grouped

def clean_text_lines(block_text):
    lines = block_text.split('\n')
    cleaned = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        line = line.strip()
        if not line or len(line) < 2 or re.match(r"^\d+:\d{2}$", line):  # skip timestamps
            continue

        if i+1 < len(lines):
            next_line = lines[i+1].strip()
            if 2 < len(next_line) < 40 and not re.search(r"\d+:\d+", next_line):
                combined = f"{line} – {next_line}"
                cleaned.append(combined)
                skip_next = True
                continue

        cleaned.append(line)
    return cleaned

def fuzzy_search_musicbrainz(query):
    try:
        result = musicbrainzngs.search_recordings(query=query, limit=1)
        if result['recording-list']:
            item = result['recording-list'][0]
            title = item['title']
            artist = item['artist-credit'][0]['name']
            return f"{title} – {artist}"
    except Exception:
        pass
    return query  # fallback: return as-is

def process_all_blocks(blocks, validate=False):
    grouped_output = defaultdict(list)

    for image_name, raw_text in tqdm(blocks, desc="Cleaning screenshots"):
        cleaned_lines = clean_text_lines(raw_text)
        for line in cleaned_lines:
            if "–" in line or "-" in line:
                entry = line
            else:
                entry = fuzzy_search_musicbrainz(line) if validate else line
                if validate:
                    time.sleep(1)  # avoid MusicBrainz throttling
            grouped_output[image_name].append(entry)

    return grouped_output

def write_cleaned_output(cleaned_data, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for img, songs in cleaned_data.items():
            f.write(f"--- {img} ---\n")
            for song in songs:
                f.write(song.strip() + '\n')
            f.write('\n')
    print(f"\n✅ Cleaned output saved to: {out_path}")

if __name__ == "__main__":
    print("🔍 Loading OCR blocks...")
    blocks = load_ocr_blocks(OCR_INPUT_PATH)

    print(f"📦 {len(blocks)} image blocks loaded.")
    cleaned_blocks = process_all_blocks(blocks, validate=USE_MUSICBRAINZ)

    print("💾 Writing cleaned output...")
    write_cleaned_output(cleaned_blocks, OUTPUT_CLEANED_PATH)
