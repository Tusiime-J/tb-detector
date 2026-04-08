# TB-Detect · DenseNet-121 Deployment

AI-powered Tuberculosis screening from chest X-rays.  
**99.4% accuracy · ROC-AUC 0.9983 · PR-AUC 0.9954**

---

## Project Structure

```
tb-detector/
├── app.py               # Flask API server
├── export_model.py      # Helper: export trained model from notebook
├── requirements.txt     # Python dependencies
├── Dockerfile           # Production container
├── docker-compose.yml   # Local dev orchestration
├── templates/
│   └── index.html       # Web UI
└── models/              # ← Place best_cnn.keras here
    └── best_cnn.keras   # (produced by export_model.py)
```

---

## Step 1 — Export your trained model

After running `TB_Detection_Complete.ipynb` to completion, run this from the same directory as the notebook:

```bash
python export_model.py
```

This finds `best_cnn.keras` (saved by ModelCheckpoint during training) and copies it into `tb-detector/models/best_cnn.keras`.

---

## Step 2 — Local development (no Docker)

```bash
cd tb-detector
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

---

## Step 3 — Production with Docker

```bash
cd tb-detector
docker compose up --build
```

Open http://localhost:8080

> **Note**: If the model file is absent, the app runs in **DEMO mode** — it still serves the UI and returns simulated predictions, useful for UI testing.

---

## Deploy to the cloud

### Option A — Google Cloud Run (recommended, free tier available)

```bash
# 1. Build & push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/tb-detect

# 2. Deploy (no cold-start penalty with --min-instances 1)
gcloud run deploy tb-detect \
  --image gcr.io/YOUR_PROJECT/tb-detect \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --set-env-vars MODEL_PATH=/app/models/best_cnn.keras

# Note: Mount the model via Cloud Storage volume or bake into image
```

### Option B — Railway / Render

1. Push this folder to a GitHub repo
2. Connect repo to Railway or Render
3. Set env vars: `MODEL_PATH`, `THRESHOLD`, `PORT`
4. Deploy — both services auto-detect the Dockerfile

### Option C — AWS EC2 / DigitalOcean Droplet

```bash
# On the server
git clone <your-repo>
cd tb-detector
docker compose up -d

# Expose via nginx reverse proxy on port 80/443
```

---

## API Reference

### `POST /predict`
Upload a chest X-ray for classification.

**Request**: `multipart/form-data` with field `file` (JPEG/PNG image, max 10 MB)

**Response**:
```json
{
  "label":       "Normal",
  "probability": 0.0731,
  "confidence":  92.7,
  "thumbnail":   "data:image/jpeg;base64,...",
  "demo_mode":   false
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model-info`
Returns full model metadata, dataset info, and performance metrics.

---

## Environment Variables

| Variable      | Default                      | Description                          |
|---------------|------------------------------|--------------------------------------|
| `MODEL_PATH`  | `models/best_cnn.keras`      | Path to the saved Keras model        |
| `THRESHOLD`   | `0.5`                        | TB probability threshold (0–1)       |
| `PORT`        | `8080` (Docker) / `5000`     | Server port                          |
| `FLASK_DEBUG` | `false`                      | Enable Flask debug mode              |

---

## Model Architecture

| Component            | Detail                                   |
|----------------------|------------------------------------------|
| Backbone             | DenseNet-121 (ImageNet pretrained)       |
| Fine-tuned layers    | Last 30 layers unfrozen                  |
| Head                 | GAP → BN → Dense(256) → Dropout(0.4) → Dense(64) → Dropout(0.2) → Sigmoid |
| Loss                 | Focal Loss (γ=2.0, α=0.75)              |
| Optimizer            | Adam (lr=1e-4)                           |
| Input size           | 224×224×3                               |
| Parameters           | 7,320,513                               |
| Training callbacks   | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## Performance (Test Set, n=840)

| Metric        | DenseNet-121  | XGBoost  | Random Forest |
|---------------|---------------|----------|---------------|
| Accuracy      | **99.40%**    | 98.10%   | 97.02%        |
| Precision     | **100.00%**   | —        | —             |
| Recall (TB)   | **96.40%**    | —        | —             |
| F1-Score      | **98.17%**    | —        | —             |
| ROC-AUC       | **0.9983**    | —        | —             |
| PR-AUC        | **0.9954**    | 0.9843   | 0.9787        |

---

## Dataset

[TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) — Tawsifur Rahman et al.
- **Normal**: 3,500 images
- **Tuberculosis**: 700 images
- Split: 70% train / 10% val / 20% test (stratified)

---

## Disclaimer

This tool is for **research and screening assistance only**. It does not constitute a clinical diagnosis. Always consult a qualified physician for medical decisions.
