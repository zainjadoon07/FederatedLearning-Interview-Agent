import pandas as pd
import re
import unicodedata
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE_DIR, 'FYP1Data.csv')

df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded dataset: {df.shape[0]} rows, columns: {list(df.columns)}")


def clean_text(text: str) -> str:
    """
    Cleans a single text string for NLP model training.
    Steps:
      1. Normalize unicode (convert accented/special chars to ASCII equivalents)
      2. Convert to lowercase
      3. Remove URLs
      4. Remove email addresses
      5. Remove punctuation and special characters (keep only letters, digits, spaces)
      6. Collapse multiple whitespace into a single space
      7. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 2. Lowercase
    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


df['Questions'] = df['Questions'].apply(clean_text)
df['Answers']   = df['Answers'].apply(clean_text)


before = len(df)
df = df[(df['Questions'].str.len() > 0) & (df['Answers'].str.len() > 0)]
after = len(df)
print(f"[INFO] Dropped {before - after} empty rows. Remaining rows: {after}")

OUTPUT_PATH = os.path.join(BASE_DIR, 'FYP1Data_cleaned.csv')
df.to_csv(OUTPUT_PATH, index=False)
print(f"[INFO] Cleaned dataset saved to: {OUTPUT_PATH}")


