"""
Train a logistic regression classifier on cached DINOv2 embeddings.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npz")
LABELS_FILE = os.path.join(BASE_DIR, "osv5m_filtered", "labels.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_CONFUSION_PAIRS = 15


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

    print("Training logistic regression (class_weight='balanced')...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)


    joblib.dump(clf, os.path.join(BASE_DIR, "country_classifier.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(classes, os.path.join(BASE_DIR, "classes.pkl"))

    y_pred = clf.predict(X_test)

    report_str = classification_report(y_test, y_pred, digits=3)
    print()
    print("--- Classification report ---")
    print(report_str)

    with open(os.path.join(BASE_DIR, "classification_report.txt"), "w") as f:
        f.write(report_str)

    print(f"confusion matrix saved (not printed, {len(classes)}x{len(classes)} is unreadable)")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    np.save(os.path.join(BASE_DIR, "confusion_matrix.npy"), cm)

    print()
    print(f"------- Top {TOP_N_CONFUSION_PAIRS} confusion pairs -------")
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

    with open(os.path.join(BASE_DIR, "confusion_pairs.txt"), "w") as f:
        f.write("\n".join(confusion_lines))


if __name__ == "__main__":
    main()
