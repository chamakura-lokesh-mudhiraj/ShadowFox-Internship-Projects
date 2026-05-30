# ShadowVision AI

Advanced image classification platform for internship and portfolio submission. It combines MobileNetV2 transfer learning, Streamlit UI, Grad-CAM explainability, analytics, webcam inference, and deployment-ready project structure.

## Features

- Professional dark glassmorphism Streamlit interface
- Image upload prediction with top-5 confidence bars
- MobileNetV2 transfer learning pipeline for Intel Scene Classification
- Grad-CAM explainable AI heatmap visualization
- Live webcam classification with FPS overlay
- Analytics dashboard for accuracy, loss, confusion matrix, and class distribution
- Downloadable JSON prediction reports
- Docker and Render deployment configuration

## Tech Stack

TensorFlow, Keras, MobileNetV2, OpenCV, Streamlit, Plotly, Pillow, NumPy, Scikit-learn, Matplotlib, Seaborn.

## Dataset

Recommended dataset: Intel Image Classification.

Expected folder layout:

```text
dataset/
  train/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
  validation/
  test/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run App

```bash
streamlit run app.py
```

The app uses `models/trained_model.h5` when available. In a fresh checkout, it falls back to MobileNetV2 ImageNet demo predictions so the product experience still opens.

## Training

```bash
python -m src.train --epochs 15 --batch-size 32
```

Training outputs:

- `models/trained_model.h5`
- `models/label_encoder.pkl`
- `models/training_history.pkl`
- `reports/model_report.txt`

## Evaluation

```bash
python -m src.evaluate --model models/trained_model.h5
```

## Deployment

### Streamlit Cloud

Use `app.py` as the entry point and install dependencies from `requirements.txt`.

### Render

Render configuration is included at `deployment/render.yaml`.

### Docker

```bash
docker build -f deployment/Dockerfile -t shadowvision-ai .
docker run -p 8501:8501 shadowvision-ai
```

## Project Structure

```text
assets/          UI styles and visual assets
dataset/         Intel image classification data
models/          trained model, label encoder, training history
notebooks/       experimentation notebook
reports/         evaluation outputs and generated reports
src/             training, prediction, Grad-CAM, webcam, dashboard modules
deployment/      Docker and Render deployment files
app.py           Streamlit application
```

## Future Improvements

- Add authentication for prediction history
- Store reports in a database
- Add model versioning with MLflow
- Export the trained model to TensorFlow Lite
- Add automated CI checks

## Author

ShadowVision AI internship submission.
