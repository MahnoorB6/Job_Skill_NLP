# Job Description Skill Extraction and Classification

## Project Overview

This project focuses on analyzing job descriptions using Natural Language Processing (NLP). The main objectives are to classify job descriptions into different job categories and identify relevant skills mentioned in them.

The project uses text-processing and machine-learning techniques to prepare job descriptions for further analysis and model development.

## Dataset Information

The project uses the Job Skill Set Dataset published by Batuhan Mutlu on Hugging Face.

**Dataset link:**  
https://huggingface.co/datasets/batuhanmtl/job-skill-set

The dataset contains 1,167 job records and five columns:

- `job_id`
- `category`
- `job_title`
- `job_description`
- `job_skill_set`

The dataset covers five job categories:

- Human Resources
- Finance
- Sales
- Business Development
- Information Technology

The original job-posting data is based on the LinkedIn Job Postings dataset.

## Project Structure

```text
Job_Skill_NLP/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── reports/
├── screenshots/
├── src/
│   ├── cleaning.py
│   └── preprocess.py
│
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- PyArrow
- spaCy
- NLTK
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Data Processing and Cleaning

The first phase of the project focuses on exploring, processing, and cleaning the dataset.

The cleaning pipeline includes:

1. Unicode normalization.
2. Conversion of text to lowercase using case folding.
3. Removal of email addresses, URLs, and phone numbers.
4. Removal of unnecessary formatting symbols.
5. Tokenization using spaCy.
6. Removal of punctuation and unnecessary tokens.
7. Lemmatization to convert words into their base forms.

Stop words are retained because they can affect the meaning of phrases in job descriptions.

## Installation

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

The project also uses the spaCy English language model, which is included in the requirements file.

## Running the Preprocessing Pipeline

From the main project directory, execute:

```bash
python src/preprocess.py
```

The script reads the original Parquet dataset, cleans the job descriptions, and saves the processed dataset.

## Output

The cleaned dataset is saved at:

```text
data/processed/cleaned_job_skill_dataset.parquet
```

The processed dataset contains all 1,167 original records, along with the additional `word_count` and `cleaned_description` columns.

## Project Objectives

1. Explore and understand job description data.
2. Clean and normalize unstructured job descriptions.
3. Classify job descriptions into their relevant categories.
4. Extract skills from job descriptions.
5. Prepare processed data for subsequent NLP experiments.

## Reproducibility

The original dataset is stored in `data/raw/`, while the processed dataset is stored in `data/processed/`.

The preprocessing code is available in the `src/` directory. The required dependencies are listed in `requirements.txt`.