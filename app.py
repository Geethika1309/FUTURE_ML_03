import re
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "resumes_with_skills.csv"


# ==================================================
# SKILLS DICTIONARY
# ==================================================

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


# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "Candidate ID",
        "Candidate Name",
        "Resume Text",
        "Cleaned Resume",
        "Extracted Skills",
        "Experience Years"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    text_columns = [
        "Candidate ID",
        "Candidate Name",
        "Resume Text",
        "Cleaned Resume",
        "Extracted Skills"
    ]

    for column in text_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
        )

    data["Experience Years"] = pd.to_numeric(
        data["Experience Years"],
        errors="coerce"
    ).fillna(0)

    return data


# ==================================================
# EXTRACT REQUIRED SKILLS
# ==================================================

def extract_required_skills(text):

    text = str(text).lower()
    found_skills = set()

    for skill in SKILLS_LIST:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, text):
            found_skills.add(skill)

    return found_skills


# ==================================================
# CONVERT SKILL TEXT TO A SET
# ==================================================

def parse_skills(value):

    if pd.isna(value) or not str(value).strip():
        return set()

    return {
        skill.strip().lower()
        for skill in str(value).split(",")
        if skill.strip()
    }


# ==================================================
# RANK CANDIDATES
# ==================================================

def rank_candidates(data, job_description):

    results = data.copy()

    # Get cleaned resume text
    resumes = (
        results["Cleaned Resume"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # Combine resumes with the job description
    documents = resumes + [
        job_description.lower()
    ]

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Separate resumes and job description
    resume_vectors = tfidf_matrix[:-1]
    job_vector = tfidf_matrix[-1]

    # Calculate cosine similarity
    similarities = cosine_similarity(
        resume_vectors,
        job_vector
    ).flatten()

    results["Text Similarity (%)"] = (
        similarities * 100
    ).round(2)

    # Extract skills required by the job
    required_skills = extract_required_skills(
        job_description
    )

    # Identify matched skills
    results["Matched Skills"] = results[
        "Extracted Skills"
    ].apply(
        lambda value: sorted(
            parse_skills(value).intersection(
                required_skills
            )
        )
    )

    # FIX: required_skills is a set, so difference()
    # works correctly when finding missing skills.
    results["Missing Skills"] = results[
        "Extracted Skills"
    ].apply(
        lambda value: sorted(
            required_skills.difference(
                parse_skills(value)
            )
        )
    )

    # Calculate skill match percentage
    if required_skills:

        results["Skill Match (%)"] = results[
            "Matched Skills"
        ].apply(
            lambda skills: round(
                len(skills) / len(required_skills) * 100,
                2
            )
        )

    else:
        results["Skill Match (%)"] = 0.0

    # Demonstration scoring formula
    # 60% text similarity + 40% skill match
    results["Overall Match (%)"] = (
        0.60 * results["Text Similarity (%)"]
        + 0.40 * results["Skill Match (%)"]
    ).round(2)

    # Rank candidates from highest to lowest score
    results = results.sort_values(
        by="Overall Match (%)",
        ascending=False
    ).reset_index(drop=True)

    results.insert(
        0,
        "Rank",
        range(1, len(results) + 1)
    )

    return results, required_skills


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("📄 Resume Screening & Candidate Ranking")

st.write(
    "Compare resumes with a job description, "
    "identify matching skills, and explore candidate rankings."
)

st.info(
    "This educational demonstration uses fictional sample resumes. "
    "Scores measure text similarity and detected keyword matches; "
    "they are not validated measures of candidate quality."
)


# ==================================================
# LOAD DATA SAFELY
# ==================================================

try:
    df = load_data()

except Exception as error:

    st.error(f"Unable to load the dataset: {error}")

    st.write(
        "Check that this file exists in your project:"
    )

    st.code("data/resumes_with_skills.csv")

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Project Information")

st.sidebar.write(
    "**Project:** Resume Screening System"
)

st.sidebar.write(
    "**Method:** TF-IDF + Cosine Similarity"
)

st.sidebar.write(
    "**Skill Extraction:** Keyword matching"
)

st.sidebar.write(
    f"**Sample resumes:** {len(df)}"
)

st.sidebar.divider()

st.sidebar.subheader("How Scoring Works")

st.sidebar.write("60% — Text similarity")
st.sidebar.write("40% — Detected skill match")

st.sidebar.caption(
    "These weights are demonstration choices, "
    "not a validated hiring assessment."
)


# ==================================================
# JOB DESCRIPTION INPUT
# ==================================================

st.subheader("1. Enter Job Description")

default_job = (
    "We are looking for a Python Developer with Python, SQL, "
    "Flask, REST APIs, Git, Pandas, and API development skills. "
    "Experience with database development and unit testing is preferred."
)

job_description = st.text_area(
    "Paste the job description below",
    value=default_job,
    height=180,
    placeholder="Enter job responsibilities and required skills...",
    key="job_description_input"
)

analyze_button = st.button(
    "🔍 Analyze and Rank Resumes",
    type="primary",
    use_container_width=True
)


# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = None

if "required_skills" not in st.session_state:
    st.session_state.required_skills = set()

if "analyzed_job_description" not in st.session_state:
    st.session_state.analyzed_job_description = ""


# ==================================================
# ANALYZE JOB DESCRIPTION
# ==================================================

if analyze_button:

    if not job_description.strip():

        st.warning(
            "Please enter a job description."
        )

    else:

        try:

            with st.spinner("Analyzing resumes..."):

                ranked_df, required_skills = rank_candidates(
                    df,
                    job_description.strip()
                )

            # Save results across Streamlit reruns
            st.session_state.ranked_df = ranked_df

            st.session_state.required_skills = (
                required_skills
            )

            st.session_state.analyzed_job_description = (
                job_description.strip()
            )

            st.success(
                "Resume analysis completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Unable to analyze resumes: {error}"
            )


# ==================================================
# DISPLAY SAVED RESULTS
# ==================================================

if st.session_state.ranked_df is not None:

    ranked_df = st.session_state.ranked_df

    required_skills = (
        st.session_state.required_skills
    )

    # ----------------------------------------------
    # CANDIDATE OVERVIEW
    # ----------------------------------------------

    st.subheader("2. Candidate Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Resumes Analyzed",
        len(ranked_df)
    )

    col2.metric(
        "Required Skills Detected",
        len(required_skills)
    )

    col3.metric(
        "Highest Text Similarity",
        f"{ranked_df['Text Similarity (%)'].max():.2f}%"
    )

    col4.metric(
        "Highest Overall Match",
        f"{ranked_df['Overall Match (%)'].max():.2f}%"
    )

    if required_skills:

        st.write(
            "**Detected job skills:**",
            ", ".join(sorted(required_skills))
        )

    else:

        st.warning(
            "No known skills were detected in the job description."
        )

    # ----------------------------------------------
    # FILTER CANDIDATES
    # ----------------------------------------------

    st.subheader("3. Filter Candidates")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        min_score = st.slider(
            "Minimum overall match score (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key="min_score_filter"
        )

    with filter_col2:

        name_search = st.text_input(
            "Search by candidate name",
            placeholder="Type a candidate name...",
            key="candidate_name_search"
        )

    # Filter by minimum overall score
    filtered_df = ranked_df[
        ranked_df["Overall Match (%)"] >= min_score
    ].copy()

    # Filter by candidate name
    if name_search.strip():

        filtered_df = filtered_df[
            filtered_df["Candidate Name"].str.contains(
                name_search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.write(
        f"Showing **{len(filtered_df)}** of "
        f"**{len(ranked_df)}** candidates."
    )

    # ----------------------------------------------
    # RANKED CANDIDATES TABLE
    # ----------------------------------------------

    st.subheader("4. Ranked Candidates")

    display_columns = [
        "Rank",
        "Candidate ID",
        "Candidate Name",
        "Experience Years",
        "Text Similarity (%)",
        "Skill Match (%)",
        "Overall Match (%)",
        "Matched Skills",
        "Missing Skills"
    ]

    display_df = filtered_df[
        display_columns
    ].copy()

    # Convert skill lists to readable strings
    for column in [
        "Matched Skills",
        "Missing Skills"
    ]:

        display_df[column] = display_df[column].apply(
            lambda skills: (
                ", ".join(map(str, skills))
                if isinstance(skills, (list, tuple, set))
                and len(skills) > 0
                else "None detected"
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------------
    # DOWNLOAD RESULTS
    # ----------------------------------------------

    st.subheader("5. Download Results")

    full_csv = ranked_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Full Ranking as CSV",
        data=full_csv,
        file_name="ranked_candidates.csv",
        mime="text/csv",
        key="download_full_ranking"
    )

    filtered_csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Candidates",
        data=filtered_csv,
        file_name="filtered_candidates.csv",
        mime="text/csv",
        key="download_filtered_candidates"
    )

    # ----------------------------------------------
    # CHARTS
    # ----------------------------------------------

    st.subheader("6. Candidate Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.write(
            "**Top 10 candidates by overall match**"
        )

        top_10 = ranked_df.head(10).sort_values(
            by="Overall Match (%)",
            ascending=True
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.barh(
            top_10["Candidate Name"],
            top_10["Overall Match (%)"]
        )

        ax.set_xlabel("Overall Match (%)")
        ax.set_ylabel("Candidate")
        ax.set_title("Top 10 Candidate Match Scores")

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with chart_col2:

        st.write(
            "**Text similarity vs. skill match**"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            ranked_df["Text Similarity (%)"],
            ranked_df["Skill Match (%)"]
        )

        ax.set_xlabel("Text Similarity (%)")
        ax.set_ylabel("Skill Match (%)")
        ax.set_title(
            "Text Similarity vs. Skill Match"
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # ----------------------------------------------
    # INDIVIDUAL CANDIDATE DETAILS
    # ----------------------------------------------

    st.subheader("7. Inspect an Individual Candidate")

    candidate_options = ranked_df[
        ["Candidate ID", "Candidate Name"]
    ].apply(
        lambda row: (
            f"{row['Candidate ID']} — "
            f"{row['Candidate Name']}"
        ),
        axis=1
    ).tolist()

    selected_candidate = st.selectbox(
        "Select a candidate",
        candidate_options,
        key="selected_candidate"
    )

    selected_id = selected_candidate.split(
        " — ",
        maxsplit=1
    )[0]

    candidate_row = ranked_df[
        ranked_df["Candidate ID"] == selected_id
    ].iloc[0]

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.write(
            "**Candidate:**",
            candidate_row["Candidate Name"]
        )

        st.write(
            "**Experience:**",
            f"{candidate_row['Experience Years']} years"
        )

        st.write(
            "**Text similarity:**",
            f"{candidate_row['Text Similarity (%)']:.2f}%"
        )

        st.write(
            "**Skill match:**",
            f"{candidate_row['Skill Match (%)']:.2f}%"
        )

        st.write(
            "**Overall match:**",
            f"{candidate_row['Overall Match (%)']:.2f}%"
        )

    with detail_col2:

        st.write("**Detected resume skills**")

        st.write(
            candidate_row["Extracted Skills"]
            or "None detected"
        )

        st.write("**Matched required skills**")

        matched_skills = candidate_row["Matched Skills"]

        st.write(
            ", ".join(matched_skills)
            if matched_skills
            else "None detected"
        )

        st.write("**Missing required skills**")

        missing_skills = candidate_row["Missing Skills"]

        st.write(
            ", ".join(missing_skills)
            if missing_skills
            else "None detected"
        )

    with st.expander("View original resume text"):

        st.write(
            candidate_row["Resume Text"]
            or "No resume text available."
        )

    st.caption(
        "A missing skill means the keyword was not detected. "
        "It does not prove that a candidate lacks that skill. "
        "Review resumes individually and do not use these scores "
        "as the sole basis for hiring decisions."
    )

else:

    st.info(
        "Enter a job description and click "
        "'Analyze and Rank Resumes' to view the results."
    )
