import io
import os
import uuid
import logging
import threading

import numpy as np
import librosa
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deepguard")

# ======================= CONFIG =======================
# Image: average of several models (ensemble). A model that fails to load is skipped.
IMAGE_MODELS = [
    "dima806/deepfake_vs_real_image_detection",
    "Organika/sdxl-detector",
]
AUDIO_MODELS = [
    "MelodyMachine/Deepfake-audio-detection-V2",
]

FAKE_THRESHOLD = 0.5          # probability above this = Fake
MAX_FILE_MB = 25
MAX_IMAGE_SIDE = 1024         # downscale very large images (speed + memory)
MAX_AUDIO_SECONDS = 30
UPLOAD_FOLDER = "uploads"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4"}

FAKE_LABELS = {"fake", "deepfake", "artificial", "ai", "ai-generated", "generated",
               "synthetic", "spoof", "spoofed"}
REAL_LABELS = {"real", "realism", "human", "authentic", "genuine", "bonafide", "bona-fide"}
# ======================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024
CORS(app)


class ModelHolder:
    """Loads the model on the first request. If loading fails, it is not retried."""

    def __init__(self, task, model_id):
        self.task = task
        self.model_id = model_id
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def short_name(self):
        return self.model_id.split("/")[-1]

    def get(self):
        with self._lock:
            if self._failed:
                return None
            if self._pipe is None:
                try:
                    log.info("Loading model: %s", self.model_id)
                    self._pipe = pipeline(self.task, model=self.model_id)
                except Exception:
                    log.exception("Model failed to load: %s", self.model_id)
                    self._failed = True
                    return None
            return self._pipe


IMAGE_HOLDERS = [ModelHolder("image-classification", m) for m in IMAGE_MODELS]
AUDIO_HOLDERS = [ModelHolder("audio-classification", m) for m in AUDIO_MODELS]


def fake_probability(results):
    """Computes the fake probability (0..1) from the model's output labels."""
    fake = real = 0.0
    for r in results:
        label = str(r["label"]).strip().lower()
        if label in FAKE_LABELS:
            fake += r["score"]
        elif label in REAL_LABELS:
            real += r["score"]
    total = fake + real
    if total == 0:
        raise ValueError(f"Unrecognized model labels: {[r['label'] for r in results]}")
    return float(fake / total)


def run_ensemble(holders, data):
    probs, parts = [], []
    for h in holders:
        pipe = h.get()
        if pipe is None:
            continue
        try:
            results = pipe(data, top_k=None)
            p = fake_probability(results)
            probs.append(p)
            parts.append(f"{h.short_name}: {p * 100:.0f}% fake")
        except Exception:
            log.exception("Prediction failed: %s", h.model_id)
    if not probs:
        raise RuntimeError("No model could be loaded or run. Check your internet connection and torch installation.")
    return float(np.mean(probs)), " | ".join(parts)


def build_response(prob, fake_text, real_text, details, model_name):
    is_fake = prob > FAKE_THRESHOLD
    conf = prob if is_fake else 1 - prob
    return {
        "verdict": fake_text if is_fake else real_text,
        "is_fake": bool(is_fake),
        "confidence": int(round(conf * 100)),
        "fake_probability": round(prob * 100, 2),
        "real_probability": round((1 - prob) * 100, 2),
        "details": details,
        "model": model_name,
    }


def analyze_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read()))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    prob, details = run_ensemble(IMAGE_HOLDERS, img)
    return build_response(prob, "Synthetic / AI-Generated", "Authentic / Real",
                          details, f"DeepGuard Image Ensemble ({len(IMAGE_HOLDERS)} models)")


def analyze_audio(path):
    y, sr = librosa.load(path, sr=16000, mono=True, duration=MAX_AUDIO_SECONDS)
    if y.size < sr // 2:
        raise ValueError("Audio is too short or empty (at least 1 second is required).")
    prob, details = run_ensemble(AUDIO_HOLDERS, {"raw": y.astype(np.float32), "sampling_rate": sr})
    return build_response(prob, "Synthetic / Cloned Voice", "Authentic Human Voice",
                          details, "DeepGuard Audio (Wav2Vec2)")


def handle_upload(kind=None):
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if kind is None:
        kind = "audio" if ext in AUDIO_EXT else "image"

    if ext not in (AUDIO_EXT if kind == "audio" else IMAGE_EXT):
        return jsonify({"error": f"Unsupported file type: {ext or 'unknown'}"}), 400

    try:
        if kind == "image":
            return jsonify(analyze_image(file))

        # Audio: save to a temporary file with a random name, delete afterwards
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
        file.save(path)
        try:
            return jsonify(analyze_audio(path))
        finally:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        log.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "active"})


@app.route("/detect", methods=["POST"])
@app.route("/analyze", methods=["POST"])
def detect():
    return handle_upload()


@app.route("/api/analyze", methods=["POST"])
def api_image():
    return handle_upload("image")


@app.route("/api/analyze-audio", methods=["POST"])
def api_audio():
    return handle_upload("audio")


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File is larger than {MAX_FILE_MB}MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
    