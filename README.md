# Resume Screening and Candidate Ranking System

## About the Project

The Resume Screening and Candidate Ranking System is a Python-based application that compares resume text with a job description.

It identifies relevant skills, calculates text similarity, highlights missing keywords, and ranks sample resumes according to their match scores.

The project demonstrates how Natural Language Processing (NLP) techniques can support resume analysis.

## Features

- Compare resumes against a job description.
- Extract skills using keyword matching.
- Calculate text similarity using TF-IDF and cosine similarity.
- Calculate a combined demonstration match score.
- Identify detected matching and missing skills.
- Rank candidates by match score.
- Filter candidates by score and name.
- Display interactive charts.
- Inspect individual candidate information.
- Download ranking results as CSV files.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- TF-IDF
- Cosine Similarity

## Project Structure

```text
future_ml_03/
├── app.py
├── prepare_data.py
├── skill_extractor.py
├── rank_candidates.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── future_ml_task3_sample_resumes.csv
│   ├── prepared_resumes.csv
│   └── resumes_with_skills.csv
├── outputs/
└── models/
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd future_ml_03
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local address displayed in your terminal.

## How to Use

1. Launch the application.
2. Enter a job description.
3. Click Analyze and Rank Resumes.
4. Review the candidate ranking table.
5. Filter candidates by score or name.
6. Inspect matching and missing skills.
7. Explore the charts.
8. Download the ranking results.

## How It Works

**TF-IDF:** Converts text into numerical representations.

**Cosine Similarity:** Measures the similarity between each resume and the job description.

**Keyword Matching:** Identifies skills listed in a predefined skill dictionary.

**Combined Demonstration Score:** Combines 60% text similarity and 40% detected skill match.

The scoring weights are illustrative and have not been validated as a hiring assessment.

## Dataset

This project uses 50 fictional sample candidate profiles for educational and demonstration purposes.

The dataset does not represent real applicants or establish real-world hiring performance.

## Limitations and Responsible Use

- The skill extractor only recognizes predefined keywords.
- Relevant skills may be missed when different wording is used.
- A missing keyword does not prove that a candidate lacks a skill.
- Text similarity does not measure actual job performance.
- The scores have not been validated for recruitment decisions.

This application is intended as a learning project and decision-support demonstration, not as an automated hiring decision-maker.

## Future Improvements

- Support PDF resume uploads.
- Improve skill extraction using NLP libraries such as spaCy.
- Support larger resume collections.
- Add more advanced semantic text matching.
- Improve the interface and reporting features.

## Author

Developed as a machine learning and NLP learning project.