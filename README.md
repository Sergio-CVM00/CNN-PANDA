# CNN PANDA

> Histopathology image-classification pipeline for prostate cancer grade assessment.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This project prepares whole-slide images from the [PANDA challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment), trains a convolutional classifier on extracted patches, and evaluates predictions across benign tissue and Gleason-pattern classes.

The code was developed as an experimental research pipeline. It expects the PANDA dataset to be downloaded separately and does not include patient images, masks, trained weights, or generated patches.

## Pipeline

1. Extract labelled image patches from whole-slide TIFF images and masks.
2. Build train, validation, and test metadata splits.
3. Train a DenseNet121 classifier with conditional augmentation and focal loss.
4. Evaluate Cohen's kappa and F1 score.
5. Produce confusion matrices, ROC curves, and training-metric visualizations.

## Repository structure

| Script | Purpose |
| --- | --- |
| `procesamiento_imagenes.py` | Extract labelled patches from PANDA slides and masks |
| `balanceo.py` | Create balanced train, validation, and test splits |
| `00_train.py` | Train and checkpoint the DenseNet121 classifier |
| `00_test.py` | Evaluate the trained model |
| `00_CMatrixROC.py` | Generate confusion matrices, ROC curves, and reports |
| `00_visualize_metrics.py` | Plot metrics recorded during training |

## Setup

### 1. Obtain the data

Download the PANDA competition dataset from Kaggle. Use of the dataset is governed by its own competition rules and license.

Expected inputs include:

- `train.csv`
- `train_images/*.tiff`
- `train_label_masks/*.tiff`

### 2. Install dependencies

Python 3.10 or 3.11 is recommended for broad TensorFlow compatibility.

```bash
git clone https://github.com/Sergio-CVM00/CNN-PANDA.git
cd CNN-PANDA

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. Configure paths

The scripts currently expose dataset and output paths as constants near the top of each file. Before running them, replace the example Windows paths with paths on your machine:

- `csv_path`, `image_folder`, `mask_folder`, and `output_folder` in `procesamiento_imagenes.py`
- `IMAGE_DIR`, `TRAIN_CSV_PATH`, and `VAL_CSV_PATH` in `00_train.py`
- Model, metadata, and output paths in the evaluation scripts

Generated datasets, patches, model weights, and reports are intentionally excluded from version control.

### 4. Run the workflow

```bash
python procesamiento_imagenes.py
python balanceo.py
python 00_train.py
python 00_test.py
python 00_CMatrixROC.py
python 00_visualize_metrics.py
```

Each stage depends on artifacts produced or configured in the preceding stages. Training requires substantially more compute and storage than the static checks below.

## Static verification

```bash
python -m compileall .
```

## Responsible use

This repository is for research and education. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## License

The source code is available under the [MIT License](./LICENSE). The PANDA dataset is not covered by this license.
