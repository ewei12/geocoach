"""
Train an MLP classifier on cached DINOv3 embeddings.

This script writes:
    classification_report_mlp.txt (full precision/recall/f1 report)
    confusion_matrix_mlp.npy (raw confusion matrix)
    confusion_pairs_mlp.txt (top 15 confusion pairs)
    country_classifier_mlp.pkl (trained MLP model)
    scaler_mlp.pkl (StandardScaler fit fresh on this run's data, not reused from train_logreg.py)
    classes_mlp.pkl (sorted list of country codes, matches clf.classes_)
"""
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings_dinov3.npz")
LABELS_FILE = os.path.join(BASE_DIR, "osv5m_filtered", "labels_dinov3.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42  # same seed as train_logreg.py so the train/test split is identical
TOP_N_CONFUSION_PAIRS = 15

# ----------------------------------------MLP architecture----------------------------------------
HIDDEN_LAYER_SIZES = (512,)
ALPHA = 1e-3
MAX_ITER = 300
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 10

def main():
    data = np.load(EMBEDDINGS_FILE)
    ids = data["ids"].astype(str)
    X_all = data["embeddings"]
    y_all = data["countries"]

    print(f"Loaded {len(X_all)} embeddings, dim {X_all.shape[1]}")

    labels_df = pd.read_csv(LABELS_FILE, low_memory=False, dtype={"id": str})
    id_to_seq = dict(zip(labels_df["id"], labels_df["sequence"]))
    sequences = np.array([id_to_seq.get(i, f"__missing_{i}") for i in ids])

    classes = sorted(set(y_all))
    print(f"Classes: {len(classes)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=sequences))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    train_seqs = set(sequences[train_idx])
    test_seqs = set(sequences[test_idx])
    overlap = train_seqs & test_seqs
    print(f"Sequence overlap between train/test: {len(overlap)} (should be 0)")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training MLP (hidden_layer_sizes={HIDDEN_LAYER_SIZES}, alpha={ALPHA}, "
          f"early_stopping={EARLY_STOPPING})...")

    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        alpha=ALPHA,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, os.path.join(BASE_DIR, "country_classifier_mlp.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler_mlp.pkl"))
    joblib.dump(classes, os.path.join(BASE_DIR, "classes_mlp.pkl"))

    y_pred = clf.predict(X_test)

    report_str = classification_report(y_test, y_pred, digits=3)
    print()
    print("--- Classification report (MLP, DINOv3 embeddings) ---")
    print(report_str)

    with open(os.path.join(BASE_DIR, "classification_report_mlp.txt"), "w") as f:
        f.write(report_str)

    print(f"confusion matrix saved (not printed, {len(classes)}x{len(classes)} is unreadable)")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    np.save(os.path.join(BASE_DIR, "confusion_matrix_mlp.npy"), cm)

    print()
    print(f"------- Top {TOP_N_CONFUSION_PAIRS} confusion pairs (MLP) -------")
    pairs = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i][j] > 0:
                pairs.append((true_cls, pred_cls, cm[i][j]))
    pairs.sort(key=lambda x: -x[2])

    confusion_lines = []
    for true_cls, pred_cls, count in pairs[:TOP_N_CONFUSION_PAIRS]:
        line = f"  {true_cls} misclassified as {pred_cls}: {count} times"
        print(line)
        confusion_lines.append(line)

    with open(os.path.join(BASE_DIR, "confusion_pairs_mlp.txt"), "w") as f:
        f.write("\n".join(confusion_lines))

if __name__ == "__main__":
    main()