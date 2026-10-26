import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Input and output files
INPUT_FILE = BASE_DIR / "data" / "resumes_with_skills.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "ranked_candidates.csv"

# Skills the system knows how to detect
SKILLS_LIST = [
    "python", "java", "c++", "sql", "mysql", "mongodb",
    "flask", "django", "fastapi", "rest apis", "git",
    "github", "html", "css", "javascript", "react",
    "node.js", "pandas", "numpy", "matplotlib", "seaborn",
    "scikit-learn", "machine learning", "deep learning",
    "natural language processing", "nlp", "data analysis",
    "data visualization", "statistics", "excel", "power bi",
    "tableau", "tensorflow", "pytorch", "xgboost", "tf-idf",
    "text classification", "api development", "unit testing",
    "docker", "linux", "aws", "communication", "problem solving"
]


def extract_required_skills(text):
    """Find known skills explicitly mentioned in the job description."""

    text = str(text).lower()
    required = []

    for skill in SKILLS_LIST:
        if skill in text:
            required.append(skill)

    return sorted(set(required))


def main():
    # Read the prepared resume dataset
    df = pd.read_csv(INPUT_FILE)

    # Ask the user for a job description
    print("\n===== RESUME SCREENING SYSTEM =====")

    job_description = input(
        "\nEnter the job description:\n"
    ).strip()

    if not job_description:
        print("Error: Please enter a job description.")
        return

    # Convert resume text and job description into TF-IDF vectors
    documents = df["Cleaned Resume"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    all_documents = documents + [job_description.lower()]
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # Last row represents the job description
    job_vector = tfidf_matrix[-1]

    # All other rows represent resumes
    resume_vectors = tfidf_matrix[:-1]

    # Calculate cosine similarity
    scores = cosine_similarity(
        resume_vectors,
        job_vector
    ).flatten()

    # Store similarity scores
    df["Text Similarity (%)"] = (scores * 100).round(2)

    # Extract required skills from the job description
    required_skills = extract_required_skills(job_description)

    # Convert extracted skills into sets for comparison
    def get_skill_set(value):
        if pd.isna(value) or not str(value).strip():
            return set()

        return {
            skill.strip().lower()
            for skill in str(value).split(",")
            if skill.strip()
        }

    df["Matched Skills"] = df["Extracted Skills"].apply(
        lambda value: sorted(
            get_skill_set(value).intersection(required_skills)
        )
    )

    df["Missing Skills"] = df["Extracted Skills"].apply(
        lambda value: sorted(
            set(required_skills).difference(get_skill_set(value))
        )
    )

    # Calculate the percentage of required skills detected
    if required_skills:
        df["Skill Match (%)"] = df["Matched Skills"].apply(
            lambda skills: round(
                len(skills) / len(required_skills) * 100, 2
            )
        )
    else:
        df["Skill Match (%)"] = 0.0

    # Combine text similarity and skill match into an illustrative score
    # This is a configurable demo formula, not a validated hiring metric.
    df["Overall Match (%)"] = (
        0.6 * df["Text Similarity (%)"]
        + 0.4 * df["Skill Match (%)"]
    ).round(2)

    # Rank candidates by the combined score
    df = df.sort_values(
        by="Overall Match (%)",
        ascending=False
    ).reset_index(drop=True)

    df.insert(0, "Rank", range(1, len(df) + 1))

    # Make skill lists easier to read in the CSV
    for column in ["Matched Skills", "Missing Skills"]:
        df[column] = df[column].apply(
            lambda skills: ", ".join(skills) if skills else "None detected"
        )

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Display top 10 candidates
    print("\n===== JOB DESCRIPTION ANALYSIS =====")
    print("Required skills:", ", ".join(required_skills)
          if required_skills else "No known skills detected")

    print("\n===== TOP 10 CANDIDATES =====")

    columns_to_show = [
        "Rank",
        "Candidate Name",
        "Text Similarity (%)",
        "Skill Match (%)",
        "Overall Match (%)",
        "Matched Skills",
        "Missing Skills"
    ]

    print(
        df[columns_to_show].head(10).to_string(index=False)
    )

    print("\nFull ranking saved to:")
    print(OUTPUT_FILE)

    print(
        "\nNote: These are text and keyword matching scores, "
        "not validated measures of candidate quality."
    )


if __name__ == "__main__":
    main()