import os
import pytesseract
from PIL import Image

# === USER CONFIGURATION ===
# Replace the values below with your actual folder and file paths
INPUT_FOLDER = r"C:\path\to\screenshots"       # Folder containing .png screenshots
OUTPUT_FILE = r"C:\path\to\OCR.txt"            # Output file for extracted text
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Change if installed elsewhere
# ==========================

def ocr_screenshots(input_folder, output_file):
    """
    Extract text from all .png images in the input folder and save to output_file.
    """
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    song_texts = []

    # List all .png files
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)
                song_texts.append(f"--- {filename} ---\n{text.strip()}\n")
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Write all extracted text to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in song_texts:
            f.write(text + '\n')

    print(f"\n✅ Finished OCR. Output saved to: {output_file}")

if __name__ == "__main__":
    ocr_screenshots(INPUT_FOLDER, OUTPUT_FILE)
