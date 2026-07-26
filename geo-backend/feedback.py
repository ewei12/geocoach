# feedback.py

# Feedback panel for when the user finds an incorrect guess to a country.
import os
import pickle
from datetime import datetime, timezone
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORRECTIONS_PATH = os.path.join(BASE_DIR, "corrections.pkl")


def log_correction(embedding: np.ndarray, correct_code: str, predicted_code: str, image_path: str = None):
    """Append one correction to corrections.pkl. Stores the raw (unscaled)
    embedding so retrain.py can re-fit the scaler + classifier from scratch."""
    if os.path.exists(CORRECTIONS_PATH):
        with open(CORRECTIONS_PATH, "rb") as f:
            corrections = pickle.load(f)
    else:
        corrections = []

    corrections.append({
        "embedding": embedding,
        "correct_code": correct_code,
        "predicted_code": predicted_code,
        "image_path": image_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    with open(CORRECTIONS_PATH, "wb") as f:
        pickle.dump(corrections, f)

    print(f"Logged correction: predicted {predicted_code}, actually {correct_code} "
          f"({len(corrections)} total corrections stored)")

    return len(corrections)


def load_corrections():
    if not os.path.exists(CORRECTIONS_PATH):
        return []
    with open(CORRECTIONS_PATH, "rb") as f:
        return pickle.load(f)