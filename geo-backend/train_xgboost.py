"""
Train an XGBoost classifier on cached DINOv3 embeddings (alternative
to training a logistic regression classifier in train_logreg.py).

Utilizes the same embeddings_dinov3.npz/labels_dinov3.csv and the same
GroupShuffleSplit-by-sequence logic, so results are directly comparable
to the logreg run.

predict.py has a MODEL_BACKEND = "xgboost" branch that expects
country_model.pkl + label_encoder.pkl. This script produces those two files.

This script writes:
    classification_report_xgboost.txt (full precision/recall/f1 report)
    confusion_matrix_xgboost.npy (raw confusion matrix)
    confusion_pairs_xgboost.txt (top 15 confusion pairs)
    country_model.pkl (trained XGBoost model)
    label_encoder.pkl (maps predictions back to country codes)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings_dinov3.npz")
LABELS_FILE = os.path.join(BASE_DIR, "osv5m_filtered", "labels_dinov3.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_CONFUSION_PAIRS = 15


def main():
    data = np.load(EMBEDDINGS_FILE)
    ids = data["ids"].astype(str)
    X_all = data["embeddings"]
    y_all_raw = data["countries"]

    print(f"Loaded {len(X_all)} embeddings, dim {X_all.shape[1]}")

    labels_df = pd.read_csv(LABELS_FILE, low_memory=False, dtype={"id": str})
    id_to_seq = dict(zip(labels_df["id"], labels_df["sequence"]))
    sequences = np.array([id_to_seq.get(i, f"__missing_{i}") for i in ids])

    # XGBoost needs integer labels, not string country codes: 
    # label_encoder.pkl is what predict.py's xgboost branch loads to map
    # predictions back to country codes.
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all_raw)
    classes = label_encoder.classes_
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

    print("Training XGBoost classifier...")
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=15,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10,
    )
    print(f"Best iteration: {clf.best_iteration}")

    joblib.dump(clf, os.path.join(BASE_DIR, "country_model.pkl"))
    joblib.dump(label_encoder, os.path.join(BASE_DIR, "label_encoder.pkl"))

    y_pred = clf.predict(X_test)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    report_str = classification_report(y_test_labels, y_pred_labels, digits=3)
    print()
    print("--- Classification report (XGBoost, DINOv3 embeddings) ---")
    print(report_str)

    with open(os.path.join(BASE_DIR, "classification_report_xgboost.txt"), "w") as f:
        f.write(report_str)

    print(f"confusion matrix saved (not printed, {len(classes)}x{len(classes)} is unreadable)")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=classes)
    np.save(os.path.join(BASE_DIR, "confusion_matrix_xgboost.npy"), cm)

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

    with open(os.path.join(BASE_DIR, "confusion_pairs_xgboost.txt"), "w") as f:
        f.write("\n".join(confusion_lines))


if __name__ == "__main__":
    main()