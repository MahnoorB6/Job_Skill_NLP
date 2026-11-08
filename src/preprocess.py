
from pathlib import Path
import pandas as pd

from cleaning import clean_job_description


# Define project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train-00000-of-00001 (1).parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_job_skill_dataset.parquet"
)


def main():

    print("JOB DESCRIPTION PREPROCESSING")
    print("-----------------------------")

    # Check whether original dataset exists
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    # Load original dataset
    df = pd.read_parquet(RAW_DATA_PATH)

    print("Original records:", len(df))

    # Calculate word count
    df["word_count"] = (
        df["job_description"]
        .fillna("")
        .astype(str)
        .apply(lambda text: len(text.split()))
    )

    # Apply cleaning function
    df["cleaned_description"] = (
        df["job_description"]
        .apply(clean_job_description)
    )

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # Verify saved dataset
    verified_df = pd.read_parquet(OUTPUT_PATH)

    print("\nPREPROCESSING RESULTS")
    print("---------------------")
    print("Cleaned records:", len(df))
    print("Saved records:", len(verified_df))
    print("Missing cleaned descriptions:",
          verified_df["cleaned_description"].isna().sum())
    print("Output file exists:", OUTPUT_PATH.exists())
    print("Output location:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
