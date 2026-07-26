# predict.py
from country_facts import code_to_name

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pycountry

MODEL_NAME = "facebook/dinov2-base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_BACKEND = "logreg"  
# "logreg" or "xgboost" 

import joblib

from huggingface_hub import hf_hub_download

HF_REPO_ID = os.environ.get("HF_REPO_ID")

if MODEL_BACKEND == "xgboost":
    clf = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="country_model.pkl"))
    label_encoder = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="label_encoder.pkl"))
    scaler = None
    classes = label_encoder.classes_
else:
    clf = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="country_classifier.pkl"))
    scaler = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="scaler.pkl"))
    classes = clf.classes_

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(device)
model.eval()


def get_embedding(image: Image.Image):
    """Returns the 1536-dim embedding (CLS + mean-pooled patch tokens),
    matching exactly what extract_embeddings.py / expand_full_final.py
    wrote into embeddings.npz, and what both classifiers were trained on."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    cls_token = hidden[:, 0, :]
    patch_mean = hidden[:, 1:, :].mean(dim=1)
    combined = torch.cat([cls_token, patch_mean], dim=-1)
    return combined.cpu().numpy()


def predict_country(image: Image.Image, top_k=5):
    embedding = get_embedding(image)
    embedding_scaled = scaler.transform(embedding) if scaler is not None else embedding

    probs = clf.predict_proba(embedding_scaled)[0]
    top_idx = probs.argsort()[-top_k:][::-1]

    top_guesses = [
        {
            "country": code_to_name(classes[i]),
            "country_code": classes[i],
            "confidence": round(float(probs[i]), 4),
        }
        for i in top_idx
        if round(float(probs[i]), 4) > 0
    ]

    top_country_code = classes[int(probs.argmax())]

    return {
        "top_country": code_to_name(top_country_code),
        "top_country_code": top_country_code,
        "top_guesses": top_guesses,
        "embedding": embedding,
    }


def name_to_code(name: str) -> str:
    """Accepts a full country name (or partial/fuzzy match) and returns its alpha-2 code."""
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        raise ValueError(f"Could not resolve '{name}' to a country code")


if __name__ == "__main__":
    # usage: python predict.py (path) --correct (country)
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--correct", help="If prediction was wrong, pass the correct country code (e.g. BR)")
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    result = predict_country(image)

    print("Prediction:", result["top_country"])
    print("\nTop guesses:")
    for g in result["top_guesses"]:
        print(g["country"], round(g["confidence"] * 100, 2), "%")

    if args.correct:
        from feedback import log_correction
        correct_code = name_to_code(args.correct)
        log_correction(
            embedding=result["embedding"],
            correct_code=correct_code,
            predicted_code=result["top_country_code"],
            image_path=args.image_path,
        )