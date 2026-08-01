import os
import uuid
from dotenv import load_dotenv
load_dotenv()

import pycountry
from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_country
from vision import analyze_image
from road_markings import analyze_road_markings
from country_facts import narrow_candidates
from reasoning import generate_reasoning
from feedback import log_correction
from storage import get_pending_store

from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

PENDING = get_pending_store()

# ----- rate limiting -----
IS_PROD = os.environ.get("ENV") == "production"
RATE_LIMIT = 5
WINDOW_SECONDS = 24 * 3600
_request_log = defaultdict(list)


def is_rate_limited(ip):
    # return True #testing purposes only
    if not IS_PROD:
        return False
    now = time.time()
    _request_log[ip] = [t for t in _request_log[ip] if now - t < WINDOW_SECONDS]
    if len(_request_log[ip]) >= RATE_LIMIT:
        return True
    _request_log[ip].append(now)
    return False

@app.route("/upload", methods=["POST"])
def upload():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if is_rate_limited(ip):
        return jsonify({"error": "Demo limit reached."}), 429

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400

    t0 = time.time()
    image_bytes = file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t1 = time.time()
    print(f"[TIMING] image load: {t1 - t0:.2f}s", flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        vision_future = executor.submit(analyze_image, pil_image)
        road_future = executor.submit(analyze_road_markings, pil_image)
        predict_future = executor.submit(predict_country, pil_image)

        vision_results = vision_future.result()
        road_raw, road_interpreted = road_future.result()
        predicted = predict_future.result()
    narrowed, contributing_clues = narrow_candidates(vision_results, road_raw, model_guesses=predicted["top_guesses"])
    print("[NARROWED DEBUG] result:", narrowed, flush=True)
    print("[NARROWED DEBUG] vision_results:", vision_results, flush=True)
    print("[NARROWED DEBUG] road_raw:", road_raw, flush=True)
    print("[DEBUG] raw model_guesses:", predicted["top_guesses"], flush=True)
    reasoning = generate_reasoning(
        contributing_clues=contributing_clues,
        narrowed_countries=narrowed,
    )

    t2 = time.time()
    print(f"[TIMING] parallel block (vision+road): {t2 - t1:.2f}s", flush=True)
    print(f"[TIMING] total: {t2 - t0:.2f}s", flush=True)

    # cache the embedding against a request_id so a later correction can
    # be tied back to it without round-tripping the embedding through the browser
    request_id = str(uuid.uuid4())
    PENDING.set(request_id, {
        "embedding": predicted["embedding"],
        "predicted_code": predicted["top_country_code"],
    })

    return jsonify({
        "vision_raw": vision_results,
        "road_markings_raw": road_raw,
        "road_markings": road_interpreted,
        "possible_countries": narrowed,
        "predicted_country": predicted["top_country"],
        "predicted_guesses": predicted["top_guesses"],
        "reasoning": reasoning,
        "request_id": request_id,
    })


# for the search box
@app.route("/countries")
def countries():
    overrides = {
        "TR": "Turkey",
        "BN": "Brunei",
        "VN": "Vietnam",
    }

    result = sorted(
        [
            {
                "code": c.alpha_2,
                "name": overrides.get(c.alpha_2, c.name),
            }
            for c in pycountry.countries
        ],
        key=lambda x: x["name"],
    )
    return jsonify(result)

# supported countries for map of data
from country_facts import TRAINED_COUNTRY_CODES
@app.get("/supported-countries")
def supported_countries():
    return {
        "countries": sorted(TRAINED_COUNTRY_CODES),
        "count": len(TRAINED_COUNTRY_CODES)
    }

import json

MODEL_INFO_PATH = "models/current_version.json"


@app.get("/model-info")
def model_info():
    with open(MODEL_INFO_PATH) as f:
        info = json.load(f)

    return jsonify({
        "version": info["version"],
        "examples": info["num_examples_after"],
        "corrections": info["num_corrections_included"],
        "created": info["created"],
    })

@app.route("/confirm", methods=["POST"])
def confirm():
    """Prediction was correct — just drop it from the pending cache, nothing to log."""
    data = request.get_json(force=True)
    PENDING.pop(data.get("request_id"))
    return jsonify({"status": "confirmed"})


@app.route("/correct", methods=["POST"])
def correct():
    # Called when the user clicks "No" and picks the right country. 
    # This pulls the cached embedding back out of PENDING, pairs it with 
    # the correct country code, and calls log_correction(), which appends 
    # {embedding, correct_code, predicted_code, timestamp} to corrections.pkl. 
    # That file is eventually fed into "retrain" script to make 
    # the classifier better at the kinds of photos it's currently getting wrong.
    data = request.get_json(force=True)
    request_id = data.get("request_id")
    correct_code = data.get("correct_code")

    if not correct_code:
        return jsonify({"error": "Missing correct_code"}), 400

    entry = PENDING.pop(request_id)
    if not request_id or entry is None:
        return jsonify({"error": "Unknown or expired request_id"}), 400
    total = log_correction(
        embedding=entry["embedding"],
        correct_code=correct_code.upper(),
        predicted_code=entry["predicted_code"],
    )
    return jsonify({"status": "logged", "correct_code": correct_code.upper(), "total_corrections": total})


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)