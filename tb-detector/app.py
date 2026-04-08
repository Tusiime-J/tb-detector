"""
TB Detection API — DenseNet-121 Backend
Flask server that serves the model and exposes a /predict endpoint.
"""

import os
import io
import base64
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 224          # DenseNet-121 standard input
MODEL_PATH    = os.getenv("MODEL_PATH", "models/best_cnn.keras")
THRESHOLD     = float(os.getenv("THRESHOLD", "0.5"))   # adjustable
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB

# ── Model Loading ─────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    try:
        import tensorflow as tf

        # Register focal loss so Keras can deserialise it
        import tensorflow.keras.backend as K

        @tf.keras.utils.register_keras_serializable()
        def focal_loss_fn(gamma=2.0, alpha=0.75):
            def loss(y_true, y_pred):
                y_true  = tf.cast(y_true, tf.float32)
                y_pred  = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
                bce     = -(y_true * tf.math.log(y_pred)
                            + (1 - y_true) * tf.math.log(1 - y_pred))
                p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
                return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
            return loss

        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={"loss_fn": focal_loss_fn},
                compile=False
            )
            logger.info("✅ DenseNet-121 model loaded successfully")
    except Exception as e:
            logger.error("❌ Failed to load model: %s", e)
            #This print helps you see the error in Docker logs
            import traceback
            traceback.print_exc()
   


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize → RGB → normalise → add batch dim."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def demo_predict(img: Image.Image) -> dict:
    """Simulated prediction when the .keras file is absent."""
    import hashlib, random
    seed = int(hashlib.md5(img.tobytes()[:1024]).hexdigest(), 16) % 10000
    rng  = random.Random(seed)
    prob = round(rng.uniform(0.05, 0.95), 4)
    label = "Tuberculosis" if prob >= THRESHOLD else "Normal"
    return {
        "label":       label,
        "probability": prob,
        "confidence":  round(max(prob, 1 - prob) * 100, 1),
        "demo_mode":   True,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    # ── Validate upload ───────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a chest X-ray image."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    raw = file.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return jsonify({"error": "Could not open file. Upload a valid image (JPEG/PNG)."}), 422

    # ── Predict ───────────────────────────────────────────────────────────────
    if model is None:
        result = demo_predict(img)
    else:
        try:
            tensor = preprocess_image(img)
            prob   = float(model.predict(tensor, verbose=0)[0][0])
            label  = "Tuberculosis" if prob >= THRESHOLD else "Normal"
            result = {
                "label":       label,
                "probability": round(prob, 4),
                "confidence":  round(max(prob, 1 - prob) * 100, 1),
                "demo_mode":   False,
            }
        except Exception as exc:
            logger.error("Inference error: %s", exc, exc_info=True)
            return jsonify({"error": f"Inference failed: {str(exc)}"}), 500

    # ── Generate thumbnail ────────────────────────────────────────────────────
    try:
        thumb = img.copy().convert("RGB")
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        result["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        result["thumbnail"] = ""

    logger.info("Prediction: label=%s prob=%.4f demo=%s",
                result["label"], result.get("probability"), result.get("demo_mode", False))

    return jsonify(result)


@app.route("/model-info")
def model_info():
    return jsonify({
        "model":        "DenseNet-121 (transfer-learned, ImageNet)",
        "parameters":   "7,320,513",
        "loss":         "Focal Loss (γ=2.0, α=0.75)",
        "optimizer":    "Adam (lr=1e-4)",
        "input_size":   f"{IMG_SIZE}×{IMG_SIZE}×3",
        "classes":      ["Normal", "Tuberculosis"],
        "threshold":    THRESHOLD,
        "performance": {
            "accuracy":  "99.40%",
            "precision": "100.00%",
            "recall_tb": "96.40%",
            "f1_score":  "98.17%",
            "roc_auc":   "0.9983",
            "pr_auc":    "0.9954",
        },
        "dataset":      "TB Chest Radiography Database (Kaggle — tawsifurrahman)",
        "train_samples": 3360,
        "test_samples":   840,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting TB-Detector on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
