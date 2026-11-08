
import re
import unicodedata
import pandas as pd
import spacy


# Load the English language model
nlp = spacy.load("en_core_web_sm")


def clean_job_description(text):
    """Clean and normalize a job description."""

    if pd.isna(text):
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", str(text))

    # 2. Convert text to lowercase
    text = text.casefold()

    # 3. Remove email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " ",
        text
    )

    # 4. Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # 5. Remove phone numbers
    text = re.sub(
        r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        " ",
        text
    )

    # 6. Remove Markdown formatting
    text = re.sub(r"[*_`#~]+", " ", text)

    # 7. Remove bullet symbols
    text = re.sub(r"[·•▪►]", " ", text)

    # 8. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Apply spaCy lemmatization
    doc = nlp(text)

    cleaned_tokens = []

    for token in doc:

        if token.is_space or token.is_punct:
            continue

        if token.is_alpha:
            cleaned_tokens.append(token.lemma_.casefold())

        elif token.like_num:
            cleaned_tokens.append(token.text)

        elif any(char.isalnum() for char in token.text):
            cleaned_tokens.append(token.text.casefold())

    # 10. Join tokens
    result = " ".join(cleaned_tokens)

    # 11. Preserve possessive endings
    result = re.sub(r"\s+'s\b", "'s", result)

    return result


# Test the function independently
if __name__ == "__main__":

    sample = (
        "THE COMPANY'S EMPLOYEES ARE "
        "DEVELOPING NEW HR SOLUTIONS."
    )

    print("Original:", sample)
    print("Cleaned:", clean_job_description(sample))
