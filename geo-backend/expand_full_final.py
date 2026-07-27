#expand_full_final.py

# Full-dataset expansion across ALL osv-5M countries that meet a minimum image count.
# Explanation of some behavior:
#   - PER_COUNTRY_CAP is a TOTAL target so countries already at/above the cap are skipped entirely.
#   - MIN_DATASET_TOTAL filters out countries that don't have enough images
#     in OSV-5M, computed from a real train.csv fetch (cached locally), across every
#     country in the dataset.
#   - No raw jpegs are written to the disk
#   - Dedups against ids already in embeddings.npz so nothing is ever downloaded or embedded twice.

# Training data: OpenStreetView-5M (Astruc et al., CVPR 2024), CC-BY-SA-4.0
# https://huggingface.co/datasets/osv5m/osv5m

#usage: python expand_full_final.py

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import csv
import io
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Configuration ---------------------------------------------------------------------------------

BACKEND_ROOT = Path("/Users/emily/Documents/geocoach-backend")
OSV_ROOT = BACKEND_ROOT / "osv5m_filtered"
LABELS_CSV = OSV_ROOT / "labels.csv"
EMBEDDINGS_NPZ = BACKEND_ROOT / "embeddings.npz"

# Cached locally after first fetch so re-runs don't re-download train.csv.
TRAIN_METADATA_CACHE = OSV_ROOT / "train_metadata_cache.csv"

PER_COUNTRY_CAP = 2000      # TOTAL target per country (existing & new)
MIN_DATASET_TOTAL = 2000    # skip countries whose image total is below this

MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

CHECKPOINT_EVERY = 1000  # write to disk this often, never lose more than this amt if a crash happens

CSV_HEADER = [
    "id", "latitude", "longitude", "thumb_original_url", "country",
    "sequence", "captured_at", "lon_bin", "lat_bin", "cell", "region",
    "sub-region", "city", "land_cover", "road_index", "drive_side",
    "climate", "soil", "dist_sea", "quadtree_10_5000", "quadtree_10_25000",
    "quadtree_10_1000", "quadtree_10_50000", "quadtree_10_12500",
    "quadtree_10_500", "quadtree_10_2500", "unique_region",
    "unique_sub-region", "unique_city", "unique_country",
    "creator_username", "creator_id",
]


def get_train_metadata():
    """Downloads (once) OSV-5M's train.csv and caches it locally, indexed
    by id, so we can look up a shard-zip entry's country without needing
    the datasets library's streaming join (which is what's broken)."""
    OSV_ROOT.mkdir(parents=True, exist_ok=True)
    if TRAIN_METADATA_CACHE.exists():
        print(f"Loading cached metadata from {TRAIN_METADATA_CACHE} ...")
        df = pd.read_csv(TRAIN_METADATA_CACHE, dtype={"id": str}, low_memory=False)
    else:
        print("Fetching train.csv metadata from the Hub (one-time)...")
        path = hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type="dataset")
        df = pd.read_csv(path, dtype={"id": str}, low_memory=False)
        df.to_csv(TRAIN_METADATA_CACHE, index=False)
    return df.set_index("id")


def compute_eligible_countries(df, min_total):
    """Every country in OSV-5M with at least `min_total` images overall --
    computed from the real metadata, not a hand-picked list."""
    totals = df["country"].value_counts()
    eligible = set(totals[totals >= min_total].index)
    excluded = totals[totals < min_total]
    print(f"{len(eligible)} countries have >= {min_total} images in the full dataset.")
    if len(excluded) > 0:
        print(f"Excluding {len(excluded)} countries below the threshold "
              f"(e.g. {list(excluded.index[:10])}{'...' if len(excluded) > 10 else ''})")
    return eligible


def list_train_shards():
    print("Listing train shard zips in osv5m/osv5m ...")
    files = list_repo_files(repo_id="osv5m/osv5m", repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("images/train/") and f.endswith(".zip"))
    print(f"Found {len(shards)} shard zips.")
    return shards

def verify_shard_naming(shards, df):
    """Downloads the first shard and checks a sample of its filenames
    against train.csv's id column, confirming the '{id}.jpg' assumption
    the main loop's row_id extraction depends on. If the naming doesn't
    match, raises immediately rather than letting a full run silently
    skip every image and report 0 new embeddings with no clue why."""
    if not shards:
        return
    print("Verifying shard filename convention against train.csv ids...")
    sample_path = download_shard_with_retry(shards[0])
    if sample_path is None:
        print("  Couldn't fetch a sample shard to verify -- proceeding without the check.")
        return
    try:
        with zipfile.ZipFile(sample_path) as zf:
            sample_names = zf.namelist()[:5]
    finally:
        try:
            os.remove(sample_path)
        except OSError:
            pass

    bad = [n for n in sample_names if Path(n).stem not in df.index]
    if bad:
        raise ValueError(
            f"Shard filenames don't match train.csv ids, e.g. {bad}. "
            f"row_id extraction (Path(name).stem) in the main loop needs adjusting "
            f"before running against the full dataset."
        )
    print(f"  OK -- sample names match train.csv ids (e.g. {sample_names[:2]})")

def load_existing_state():
    """Returns (existing_ids set, per-country counts dict from labels.csv)."""
    existing_ids = set()
    if EMBEDDINGS_NPZ.exists():
        d = np.load(EMBEDDINGS_NPZ)
        existing_ids = set(d["ids"].tolist())
    print(f"Loaded {len(existing_ids)} existing embedded ids for dedup.")

    counts = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                c = row.get("country")
                if c:
                    counts[c] = counts.get(c, 0) + 1
    print(f"Existing per-country counts loaded for {len(counts)} countries.")
    return existing_ids, counts


def checkpoint(pending_rows, pending_embeddings, pending_ids, pending_countries):
    """Appends pending data to labels.csv and embeddings.npz, then clears
    it. Safe to call repeatedly -- each call only adds what's new since
    the last checkpoint, so a crash mid-run loses at most CHECKPOINT_EVERY
    images, not the whole run."""
    if not pending_rows:
        return

    file_exists = LABELS_CSV.exists()
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerows(pending_rows)

    new_embeddings = np.stack(pending_embeddings)
    new_ids = np.array(pending_ids, dtype=str)
    new_countries = np.array(pending_countries, dtype=str)

    if EMBEDDINGS_NPZ.exists():
        old = np.load(EMBEDDINGS_NPZ)
        all_embeddings = np.concatenate([old["embeddings"], new_embeddings])
        all_ids = np.concatenate([old["ids"].astype(str), new_ids])
        all_countries = np.concatenate([old["countries"].astype(str), new_countries])
    else:
        all_embeddings, all_ids, all_countries = new_embeddings, new_ids, new_countries

    np.savez(EMBEDDINGS_NPZ, embeddings=all_embeddings, ids=all_ids, countries=all_countries)
    print(f"  [checkpoint] wrote {len(pending_rows)} rows -- "
          f"embeddings.npz now has {len(all_ids)} total entries")


def download_shard_with_retry(shard_path, max_retries=5):
    """shard_path looks like 'images/train/00.zip'. Returns local cache
    path, or None if it genuinely can't be fetched (skip this shard and
    move on rather than looping forever on a persistent failure)."""
    subfolder, filename = shard_path.rsplit("/", 1)
    for attempt in range(max_retries):
        try:
            return hf_hub_download(
                repo_id="osv5m/osv5m",
                filename=filename,
                subfolder=subfolder,
                repo_type="dataset",
            )
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"  Download hiccup on {shard_path} ({e}); retrying in {wait}s "
                  f"[{attempt + 1}/{max_retries}]")
            time.sleep(wait)
    print(f"  Giving up on {shard_path} after {max_retries} attempts -- skipping shard.")
    return None


def main():
    df = get_train_metadata()
    eligible = compute_eligible_countries(df, MIN_DATASET_TOTAL)
    if not eligible:
        print("No eligible countries -- nothing to do.")
        return

    existing_ids, counts = load_existing_state()
    shards = list_train_shards()

    verify_shard_naming(shards, df)

    download_pool = ThreadPoolExecutor(max_workers=1)
    next_shard_future = download_pool.submit(download_shard_with_retry, shards[0]) if shards else None

    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    pending_rows, pending_embeddings, pending_ids, pending_countries = [], [], [], []
    batch_imgs, batch_ids, batch_countries, batch_rows = [], [], [], []
    total_new = 0
    since_checkpoint = 0

    def flush_batch():
        nonlocal batch_imgs, batch_ids, batch_countries, batch_rows
        nonlocal total_new, since_checkpoint
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            # # CLS token -- matches extract_embeddings.py exactly
            # out = model(**inputs).last_hidden_state[:, 0, :]
            hidden = model(**inputs).last_hidden_state
            cls_tokens = hidden[:, 0, :]
            patch_means = hidden[:, 1:, :].mean(dim=1)
            out = torch.cat([cls_tokens, patch_means], dim=-1)
        out = out.cpu().numpy()

        for i in range(len(batch_ids)):
            pending_embeddings.append(out[i])
            pending_ids.append(batch_ids[i])
            pending_countries.append(batch_countries[i])
            pending_rows.append(batch_rows[i])
            total_new += 1
            since_checkpoint += 1

        print(f"  ...{total_new} new images embedded so far "
              f"(latest batch: {set(batch_countries)})")

        batch_imgs, batch_ids, batch_countries, batch_rows = [], [], [], []

        if since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint(pending_rows, pending_embeddings, pending_ids, pending_countries)
            pending_rows.clear()
            pending_embeddings.clear()
            pending_ids.clear()
            pending_countries.clear()
            since_checkpoint = 0

    for i, shard_path in enumerate(shards):
        if all(counts.get(c, 0) >= PER_COUNTRY_CAP for c in eligible):
            print("All eligible countries at cap -- stopping before further shard downloads.")
            break

        print(f"Fetching shard {shard_path} ...")
        local_zip = next_shard_future.result()
        if i + 1 < len(shards):
            next_shard_future = download_pool.submit(download_shard_with_retry, shards[i + 1])
        if local_zip is None:
            continue

        try:
            with zipfile.ZipFile(local_zip, "r") as zf:
                names = zf.namelist()
                for name in names:
                    if all(counts.get(c, 0) >= PER_COUNTRY_CAP for c in eligible):
                        break

                    row_id = Path(name).stem  # assumes '{id}.jpg' -- verify against namelist()[:5]
                    if row_id in existing_ids or row_id not in df.index:
                        continue

                    country = df.at[row_id, "country"]
                    if country not in eligible or counts.get(country, 0) >= PER_COUNTRY_CAP:
                        continue

                    try:
                        img_bytes = zf.read(name)
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    except Exception as e:
                        print(f"  Skipping unreadable entry {name}: {e}")
                        continue

                    meta = df.loc[row_id]
                    csv_row = {col: "" for col in CSV_HEADER}
                    csv_row["id"] = row_id
                    csv_row["country"] = country
                    for col in ("latitude", "longitude", "sequence", "captured_at"):
                        if col in meta:
                            csv_row[col] = meta[col]

                    batch_imgs.append(img)
                    batch_ids.append(row_id)
                    batch_countries.append(country)
                    batch_rows.append(csv_row)

                    counts[country] = counts.get(country, 0) + 1
                    existing_ids.add(row_id)

                    if len(batch_imgs) >= BATCH_SIZE:
                        flush_batch()
        finally:
            # Shard zip is intentionally not kept around -- re-download later if a future pass needs it again.
            try:
                os.remove(local_zip)
            except OSError:
                pass

        remaining = sum(1 for c in eligible if counts.get(c, 0) < PER_COUNTRY_CAP)
        print(f"  {remaining}/{len(eligible)} eligible countries still under cap after this shard.")

    flush_batch()  # remaining partial batch
    checkpoint(pending_rows, pending_embeddings, pending_ids, pending_countries)  # final partial checkpoint

    print(f"\nDone. Embedded {total_new} new images total (no raw jpgs saved).")
    print("Final per-country counts:", {c: counts.get(c, 0) for c in eligible})


if __name__ == "__main__":
    main()