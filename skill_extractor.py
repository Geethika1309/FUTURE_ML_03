import pandas as pd
import re
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# File paths
INPUT_FILE = BASE_DIR / "data" / "prepared_resumes.csv"
OUTPUT_FILE = BASE_DIR / "data" / "resumes_with_skills.csv"

# Skills dictionary
SKILLS_LIST = [
    "python",
    "java",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "flask",
    "django",
    "fastapi",
    "rest apis",
    "git",
    "github",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "data analysis",
    "data visualization",
    "statistics",
    "excel",
    "power bi",
    "tableau",
    "tensorflow",
    "pytorch",
    "xgboost",
    "tf-idf",
    "text classification",
    "api development",
    "unit testing",
    "docker",
    "linux",
    "aws",
    "communication",
    "problem solving"
]


def extract_skills(text):
    """Find known skills mentioned in a piece of text."""

    text = str(text).lower()
    found_skills = []

    for skill in SKILLS_LIST:
        # Match the skill as a complete term where possible
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(set(found_skills))


def main():
    # Load prepared resumes
    df = pd.read_csv(INPUT_FILE)

    # Extract skills from the resume text
    df["Extracted Skills"] = df["Cleaned Resume"].apply(extract_skills)

    # Convert skill lists into readable strings for CSV
    df["Extracted Skills"] = df["Extracted Skills"].apply(
        lambda skills: ", ".join(skills)
    )

    # Save results
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== SKILL EXTRACTION RESULTS ==========")
    print("Total resumes processed:", len(df))

    for _, row in df.head(5).iterrows():
        print("\nCandidate:", row["Candidate Name"])
        print("Extracted Skills:", row["Extracted Skills"])

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nSkill extraction completed successfully!")


if __name__ == "__main__":
    main()