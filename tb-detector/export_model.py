"""
export_model.py — Run this from your Jupyter environment AFTER training.

It loads the trained DenseNet-121 from your notebook session and saves it
to models/best_cnn.keras so the Flask app can serve it.

Usage (inside Jupyter or from terminal after running the notebook):
    python export_model.py
"""

import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# ── Re-register focal loss ─────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce     = -(y_true * tf.math.log(y_pred)
                    + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
    return loss_fn


SOURCE_PATHS = [
    "best_cnn.keras",          # default ModelCheckpoint save location
    "../best_cnn.keras",
    "models/best_cnn.keras",
]
DEST = os.path.join(os.path.dirname(__file__), "models", "best_cnn.keras")
os.makedirs(os.path.dirname(DEST), exist_ok=True)

def loss_fn(y_true, y_pred):
    return None # We don't need the actual math for inference
def load_from_path(path: str):
    print(f"  Trying: {path} … ", end="")
    if not os.path.exists(path):
        print("not found")
        return None
    
    model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn}, compile=False)
        
    print(f"loaded  ({model.count_params():,} params)")
    return model


def main():
    model = None
    for p in SOURCE_PATHS:
        model = load_from_path(p)
        if model:
            break

    if model is None:
        print("\n❌  Could not find best_cnn.keras in any of the expected locations.")
        print("    Make sure you have run the notebook to completion first,")
        print("    then re-run this script from the same working directory.\n")
        sys.exit(1)

    # Quick smoke-test
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    prob  = model.predict(dummy, verbose=0)[0][0]
    print(f"  Smoke test: prob for blank image = {prob:.4f}  ✓")

    model.save(DEST)
    print(f"\n✅  Model saved to: {DEST}")
    print("    Copy the `models/` folder into your tb-detector/ directory")
    print("    and run:  docker compose up\n")


if __name__ == "__main__":
    main()
