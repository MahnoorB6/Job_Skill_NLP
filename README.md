# Job Description Skill Extraction and Classification

## Project Overview

This project focuses on analyzing job descriptions using Natural Language Processing (NLP). The main objectives are to classify job descriptions into different job categories and identify relevant skills mentioned in them.

The project uses text-processing and machine-learning techniques to prepare job descriptions for further analysis and model development.

## Dataset Information

The project uses the **Job Skill Set Dataset** published by Batuhan Mutlu on Hugging Face.

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

**Dataset availability:** The dataset files are not included in this repository. Download the dataset from the source link above and place the original Parquet file inside `data/raw/`. The preprocessing script generates the cleaned dataset inside `data/processed/`.

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
│   └── job_skill_analysis.ipynb
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

1. Unicode normalization using NFKC.
2. Conversion of text to lowercase using case folding.
3. Removal of email addresses, URLs, and phone numbers.
4. Removal of unnecessary formatting symbols.
5. Tokenization using spaCy.
6. Removal of punctuation and unnecessary tokens.
7. Lemmatization to convert words into their base forms.

Stop words are retained because they can affect the meaning of phrases in job descriptions. For example, removing words such as "without" could change the meaning of a sentence.

## Installation

Clone the repository:

```bash
git clone https://github.com/MahnoorB6/Job_Skill_NLP.git
cd Job_Skill_NLP
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

The project also uses the spaCy English language model, which is included in the requirements file.

## Running the Preprocessing Pipeline

Before running the preprocessing script, make sure the original dataset is available in the `data/raw/` directory.

The script expects the following filename:

```text
train-00000-of-00001 (1).parquet
```

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

The processed dataset contains all 1,167 original records, along with two additional columns:

- `word_count`
- `cleaned_description`

The original columns are also retained.

## Project Objectives

1. Explore and understand job description data.
2. Clean and normalize unstructured job descriptions.
3. Classify job descriptions into their relevant categories.
4. Extract skills from job descriptions.
5. Prepare processed data for subsequent NLP experiments.

## Reproducibility

The dataset files are not included in this repository. To reproduce the preprocessing results:

1. Download the original dataset from the Hugging Face link provided above.
2. Place the Parquet file inside `data/raw/`.
3. Make sure its filename is `train-00000-of-00001 (1).parquet`.
4. Install the required dependencies using `pip install -r requirements.txt`.
5. Run `python src/preprocess.py` from the project root directory.

The cleaned dataset will be generated in `data/processed/`.

The preprocessing code is available in the `src/` directory, and the required dependencies are listed in `requirements.txt`.

## Assignment 2 — Word Representations

### Overview

This phase implements count-based and learned word representations using the cleaned job-description dataset from Assignment 1.

### Dataset

- Documents: 1,167
- Total tokens: 613,180
- Unique words: 26,485
- Input column: `cleaned_description`

### Implemented Techniques

**Part A — Count-Based Representations**
- Bag of Words (BoW)
- TF-IDF
- Unigrams, bigrams and trigrams
- Manual TF-IDF and bigram verification

**Part B — Learned Word Embeddings**
- Word2Vec CBOW
- Word2Vec Skip-gram
- fastText
- Out-of-vocabulary word demonstration

**Part C — Evaluation**
- Most similar words
- Cosine similarity
- Word analogy
- Count-based versus learned representation comparison

**Part D — Model Packaging**
- Saved TF-IDF vectorizer
- Saved CBOW, Skip-gram and fastText models
- Reusable document-vectorization interface in `src/vectorizer.py`

### Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running Assignment 2

1. Prepare the cleaned dataset using the Assignment 1 preprocessing pipeline.
2. Open `notebooks/assignment_2_word_representations.ipynb`.
3. Run the notebook cells in order.
4. Save the fitted vectorizer and embedding models.
5. Test the reusable document-vectorization interface.

### Saved Models

The `models/` directory contains the TF-IDF vectorizer and the three trained embedding models. These models can be loaded without retraining.

### Report

The Assignment 2 report is available in:

`reports/NLP_Assignment_2_Report.pdf`

### Team Contributions

**Emaan Abbas:** Bag of Words, TF-IDF, N-grams, manual calculations, and corresponding documentation.

**Mahnoor Bhatti:** CBOW, Skip-gram, fastText, similarity experiments, analogy experiments, and corresponding documentation.

Both members contributed to integration, testing, and final submission.
