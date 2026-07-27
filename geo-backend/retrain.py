# retrain.py
#
# Loads current embeddings + whatever's in corrections.pkl, updates the
# scaler & classifier, saves as a new version under model_versions/.
#
# Versioning is additive - v1 builds on base + corrections up to that
# point, v2 builds on v1 + whatever's been logged since. corrections.pkl
# gets wiped after a successful retrain so nothing gets folded in twice.
#
# app.py/predict.py just load country_classifier.pkl / scaler.pkl, so
# those get overwritten in place - no other code needs to change.
#
# Run after ~10-20+ corrections, ideally spread across countries/photos,
# not the same shot corrected repeatedly.

import os
import re
import json
import pickle
import shutil
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from feedback import load_corrections as _load_corrections_from_store, clear_corrections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# live paths - what app.py/predict.py actually load
CORRECTIONS_PATH = os.path.join(BASE_DIR, "corrections.pkl")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "country_classifier.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npz")  # base training set, only used pre-v1

VERSIONS_DIR = os.path.join(BASE_DIR, "model_versions")
CURRENT_VERSION_FILE = os.path.join(VERSIONS_DIR, "current_version.txt")

# corrections count as 1 regular example each
CORRECTION_WEIGHT = 1.0


def load_corrections():
    corrections = _load_corrections_from_store()
    if not corrections:
        raise ValueError("No corrections found — log at least one through the feedback UI first.")
    return corrections


def load_training_data():
    """Pulls from the current version's embeddings.npz if one exists,
    otherwise the original base file. This is the additive part - v2
    builds on v1's data, not the base again.

    keys: 'embeddings' (X), 'countries' (y), 'ids' - matches
    extract_embeddings.py / expand_full_final.py.
    """
    current = get_current_version()
    if current:
        path = os.path.join(VERSIONS_DIR, current, "embeddings.npz")
        source = f"version {current}"
    else:
        path = EMBEDDINGS_PATH
        source = "original base file"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find {path} (expected {source}). This script expects an "
            ".npz file with arrays 'embeddings', 'countries', and 'ids'."
        )
    print(f"  building on top of: {source}")
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["countries"], data["ids"]


def next_version_name():
    """v1, v2, v3... based on what's already in model_versions/."""
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(VERSIONS_DIR) if re.match(r"^v(\d+)_", d)]
    nums = [int(re.match(r"^v(\d+)_", d).group(1)) for d in existing]
    n = max(nums, default=0) + 1
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"v{n}_{ts}", n


def get_current_version():
    if os.path.exists(CURRENT_VERSION_FILE):
        with open(CURRENT_VERSION_FILE) as f:
            return f.read().strip()
    return None


def main():
    print("Loading corrections...")
    corrections = load_corrections()
    print(f"  {len(corrections)} corrections found")

    by_country = {}
    for c in corrections:
        by_country[c["correct_code"]] = by_country.get(c["correct_code"], 0) + 1
    print("  breakdown by corrected country:", by_country)

    X_corr = np.vstack([np.asarray(c["embedding"]).reshape(1, -1) for c in corrections])
    y_corr = np.array([c["correct_code"] for c in corrections])
    # corrections have no generated id - use image_path, fall back to synthetic
    ids_corr = np.array([
        c.get("image_path", f"correction_{i}") for i, c in enumerate(corrections)
    ])

    print("Loading training data to build on...")
    X_orig, y_orig, ids_orig = load_training_data()
    print(f"  {len(y_orig)} existing examples")

    X_all = np.vstack([X_orig, X_corr])
    y_all = np.concatenate([y_orig, y_corr])
    ids_all = np.concatenate([ids_orig, ids_corr])

    # y_all is already cumulative (y_orig already has all prior corrections
    # baked in from the last version), so counting y_all directly gives the
    # real running total. Don't add this on top of the previous manifest's
    # counts - that double-counts everything already in y_orig.
    country_image_counts = dict(Counter(y_all.tolist()))

    weights = np.concatenate([
        np.ones(len(y_orig)),
        np.full(len(y_corr), CORRECTION_WEIGHT),
    ])

    print("Fitting scaler...")
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)

    print("Loading existing classifier to clone its type + hyperparameters...")
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Couldn't find existing classifier at {CLASSIFIER_PATH}.")
    old_clf = joblib.load(CLASSIFIER_PATH)
    print(f"  existing classifier: {type(old_clf).__name__}")
    print(f"  params: {old_clf.get_params()}")

    new_clf = clone(old_clf)

    print("Fitting new classifier...")
    try:
        new_clf.fit(X_scaled, y_all, sample_weight=weights)
        used_weights = True
    except TypeError:
        print(f"  {type(new_clf).__name__}.fit() doesn't support sample_weight — "
              f"fitting without it. Corrections count as 1 example each, not {CORRECTION_WEIGHT}x.")
        new_clf.fit(X_scaled, y_all)
        used_weights = False

    X_corr_scaled = scaler.transform(X_corr)
    preds = new_clf.predict(X_corr_scaled)
    correct_now = int((preds == y_corr).sum())
    print(f"Sanity check: retrained model gets {correct_now}/{len(y_corr)} "
          f"of the corrected examples right on the training set itself.")

    # write the new version folder
    version_name, version_num = next_version_name()
    parent_version = get_current_version()
    version_dir = os.path.join(VERSIONS_DIR, version_name)
    os.makedirs(version_dir, exist_ok=True)

    joblib.dump(new_clf, os.path.join(version_dir, "country_classifier.pkl"))
    joblib.dump(scaler, os.path.join(version_dir, "scaler.pkl"))
    np.savez(os.path.join(version_dir, "embeddings.npz"), embeddings=X_all, countries=y_all, ids=ids_all)
    # keep the exact corrections that went into this version for reference
    with open(os.path.join(version_dir, "corrections_included.pkl"), "wb") as f:
        pickle.dump(corrections, f)

    manifest = {
        "version": version_name,
        "version_number": version_num,
        "parent_version": parent_version,
        "created": datetime.now().isoformat(),
        "classifier_type": type(new_clf).__name__,
        "classifier_params": {k: str(v) for k, v in new_clf.get_params().items()},
        "num_examples_before": int(len(y_orig)),
        "num_corrections_included": int(len(y_corr)),
        "num_examples_after": int(len(y_all)),
        "corrections_breakdown": by_country,
        "country_image_counts": country_image_counts,
        "correction_weight_used": CORRECTION_WEIGHT if used_weights else 1.0,
        "sample_weight_supported": used_weights,
        "sanity_check": f"{correct_now}/{len(y_corr)} corrections classified correctly post-retrain",
        "notes": "",
    }
    with open(os.path.join(version_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # point the live files at this new version
    joblib.dump(new_clf, CLASSIFIER_PATH)  # overwrites live country_classifier.pkl
    joblib.dump(scaler, SCALER_PATH)  # overwrites live scaler.pkl
    with open(CURRENT_VERSION_FILE, "w") as f:
        f.write(version_name)  # points at the new version

    # corrections are folded in now, wipe so next retrain doesn't reapply them
    clear_corrections()
    print(f"  cleared corrections store ({len(corrections)} corrections folded into {version_name})")

    print(f"\nSaved new version: {version_name}  (parent: {parent_version or 'none'})")
    print(f"  {version_dir}/")
    print(f"    country_classifier.pkl")
    print(f"    scaler.pkl")
    print(f"    embeddings.npz          (cumulative training data - {len(y_all)} examples)")
    print(f"    corrections_included.pkl  (just the {len(y_corr)} corrections added this round)")
    print(f"    manifest.json")
    print(f"\nLive model files updated. Restart app.py to pick up {version_name}.")


def rollback(version_name):
    """Points live model files back at an older version.
    Heads up: this also changes what future retrains build on top of -
    next retrain after a rollback builds on THIS version, not whatever
    was current before.
    Usage: python retrain.py --rollback v1_2026-07-16_1430"""
    version_dir = os.path.join(VERSIONS_DIR, version_name)
    clf_path = os.path.join(version_dir, "country_classifier.pkl")
    scaler_path = os.path.join(version_dir, "scaler.pkl")

    if not os.path.exists(clf_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No complete version found at {version_dir}")

    shutil.copy(clf_path, CLASSIFIER_PATH)
    shutil.copy(scaler_path, SCALER_PATH)
    with open(CURRENT_VERSION_FILE, "w") as f:
        f.write(version_name)

    print(f"Rolled back to {version_name}. Restart app.py to pick it up.")
    print("Note: the next retrain will build on this version's embeddings going forward.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        if len(sys.argv) < 3:
            print("Usage: python retrain.py --rollback <version_name>")
            print(f"Available versions: {os.listdir(VERSIONS_DIR) if os.path.exists(VERSIONS_DIR) else 'none yet'}")
        else:
            rollback(sys.argv[2])
    else:
        main()