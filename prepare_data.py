import pandas as pd
import re
from pathlib import Path

# Get the project folder
BASE_DIR = Path(__file__).resolve().parent

# Define file paths
INPUT_FILE = BASE_DIR / "data" / "future_ml_task3_sample_resumes.csv"
OUTPUT_FILE = BASE_DIR / "data" / "prepared_resumes.csv"

# 1. Load the dataset
df = pd.read_csv(INPUT_FILE)

print("\n========== ORIGINAL DATASET ==========")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Check column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 3. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 4. Remove duplicate rows
original_rows = len(df)

df = df.drop_duplicates()

print("\nDuplicate rows removed:", original_rows - len(df))

# 5. Remove rows without resume text
df["Resume Text"] = df["Resume Text"].fillna("").astype(str)

df = df[df["Resume Text"].str.strip() != ""].copy()

# 6. Clean resume text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Cleaned Resume"] = df["Resume Text"].apply(clean_text)

# 7. Clean skills
df["Skills"] = df["Skills"].fillna("").astype(str)
df["Cleaned Skills"] = df["Skills"].apply(clean_text)

# 8. Remove duplicate candidate IDs, keeping the first record
df = df.drop_duplicates(subset=["Candidate ID"], keep="first")

# 9. Save the prepared dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== PREPARATION SUMMARY ==========")
print("Rows before cleaning:", original_rows)
print("Rows after cleaning:", len(df))
print("Columns after cleaning:", df.columns.tolist())
print("\nPrepared dataset saved to:")
print(OUTPUT_FILE)

print("\n========== CLEANED RESUME EXAMPLE ==========")
if not df.empty:
    print(df["Cleaned Resume"].iloc[0])

print("\nData preparation completed successfully!")