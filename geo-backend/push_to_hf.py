# push_to_hf.py

# Uploads the current country_classifier.pkl/scaler.pkl to HF repo

# Ran after retrain.py, which saves the new pkls locally. 

import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "country_classifier.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

HF_REPO_ID = os.environ.get("HF_REPO_ID")


def main():
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID not set. Check your .env file.")

    for path in (CLASSIFIER_PATH, SCALER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Couldn't find {path}. Run retrain.py first.")

    api = HfApi()

    print(f"Uploading to {HF_REPO_ID} ...")
    api.upload_file(
        path_or_fileobj=CLASSIFIER_PATH,
        path_in_repo="country_classifier.pkl",
        repo_id=HF_REPO_ID,
    )
    print("  uploaded country_classifier.pkl")

    api.upload_file(
        path_or_fileobj=SCALER_PATH,
        path_in_repo="scaler.pkl",
        repo_id=HF_REPO_ID,
    )
    print("  uploaded scaler.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()