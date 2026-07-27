# feedback.py
import os
import pickle
from datetime import datetime, timezone
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORRECTIONS_PATH = os.path.join(BASE_DIR, "corrections.pkl")

DATABASE_URL = os.getenv("DATABASE_URL")


def _get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def log_correction(embedding, correct_code: str, predicted_code: str, image_path: str = None):
    """Logs a correction either to Postgres (if DATABASE_URL is set) or to
    corrections.pkl locally. Stores the raw (unscaled) embedding so
    retrain.py can update the scaler & classifier."""

    if DATABASE_URL:
        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO corrections (embedding, correct_code, predicted_code, image_path)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (embedding_list, correct_code, predicted_code, image_path),
                )
                cur.execute("SELECT COUNT(*) FROM corrections")
                total = cur.fetchone()[0]

        print(f"[feedback:postgres] Logged correction: predicted {predicted_code}, actually {correct_code} "
              f"({total} total corrections stored)")
        return total

    # --- local pkl fallback ------------------------------------
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

    print(f"[feedback:local] Logged correction: predicted {predicted_code}, actually {correct_code} "
          f"({len(corrections)} total corrections stored)")
    return len(corrections)


def load_corrections():
    if DATABASE_URL:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding, correct_code, predicted_code, image_path, created_at FROM corrections"
                )
                rows = cur.fetchall()

        return [
            {
                "embedding": np.array(embedding),
                "correct_code": correct_code,
                "predicted_code": predicted_code,
                "image_path": image_path,
                "timestamp": created_at.isoformat(),
            }
            for embedding, correct_code, predicted_code, image_path, created_at in rows
        ]

    # --- local pickle fallback ------------------------------------
    if not os.path.exists(CORRECTIONS_PATH):
        return []
    with open(CORRECTIONS_PATH, "rb") as f:
        return pickle.load(f)

def clear_corrections():
    """Wipes all logged corrections after they've been folded into a retrain."""
    if DATABASE_URL:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM corrections")
        print("[feedback:postgres] Cleared corrections table")
        return

    with open(CORRECTIONS_PATH, "wb") as f:
        pickle.dump([], f)
    print("[feedback:local] Cleared corrections.pkl")