# Histopathologic Cancer Detection (PCam) — Kaggle Project

Identify metastatic cancer in histopathology image patches (96×96) using TensorFlow/Keras.  
This repo contains a Kaggle-ready notebook, robust TIFF pipelines, a baseline CNN model, optional EfficientNet transfer learning, and submission workflow.

## Overview
- **Task:** Binary image classification — 1 (metastasis present) vs 0 (absent)
- **Dataset:** PatchCamelyon (PCam) — Kaggle de-duplicated version
- **Metric:** ROC–AUC (submissions require probabilities)
- **Why PCam?** Clinically relevant, compact benchmark dataset; approachable for deep learning research.

## Repo Structure
```
.
├── notebooks/
│   └── pcam_histopathologic_cancer_detection.ipynb
├── src/                        # optional: helpers
├── models/                     # optional: saved checkpoints
├── reports/
│   └── figures/                # EDA plots, leaderboard screenshot
├── requirements.txt            # if running locally
└── README.md
```

## Data
Dataset: [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)  
Path on Kaggle: `/kaggle/input/histopathologic-cancer-detection/`
```
/train_labels.csv
/train/*.tif
/test/*.tif
```

## Usage on Kaggle
1. Open competition → **Code → New Notebook** (select GPU).
2. Attach dataset **Histopathologic Cancer Detection**.
3. Upload notebook from `notebooks/pcam_histopathologic_cancer_detection.ipynb`.
4. Run all cells. Submission file written to `/kaggle/working/pcam-output/submission.csv`.
5. Submit file via **Save Version → Submit to Competition**.

### Local Usage (optional)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PCAM_BASE_DIR="/path/to/histopathologic-cancer-detection"
jupyter notebook notebooks/pcam_histopathologic_cancer_detection.ipynb
```

## Notebook Contents
- Problem & Data summary
- EDA: class balance, histograms, sample grids
- Model building:
  - Small CNN baseline
  - EfficientNetB0 (transfer learning)
  - Robust tf.data pipeline with TIFF decoding
- Callbacks: checkpoint, ReduceLROnPlateau, EarlyStopping, CSV logs
- Evaluation: ROC–AUC, threshold search, optional Grad-CAM
- Submission: generates `submission.csv`

## Results
- **Baseline small CNN:** Validation ROC–AUC ≈ 0.947
- **Techniques that helped:** augmentation, class weights, checkpointing
- **Issues:** aggressive augmentations & high parallelism sometimes stalled I/O; tuned pipeline fixed this.

## Roadmap
- Add K-fold CV + ensembles
- Experiment with EfficientNetV2 / ConvNeXt
- Stain normalization & hard negative mining
- TTA & uncertainty estimation
- Interpretability with Grad-CAM

## Deliverables
- Jupyter Notebook (analysis + results)
- Public GitHub repository
- Kaggle leaderboard screenshot

## License
MIT License (or add your choice here)
