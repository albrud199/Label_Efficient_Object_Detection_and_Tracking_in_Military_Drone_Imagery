# Label-Efficient Object Detection and Tracking in Military Drone Imagery

![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange)

A computer-vision research project focused on improving object detection and tracking performance in military drone imagery under limited-label conditions.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Expected Use Cases](#expected-use-cases)
- [Features and Capabilities](#features-and-capabilities)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage and Workflow](#usage-and-workflow)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Suggested GitHub Repository Description](#suggested-github-repository-description)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This repository contains notebooks and supporting study artifacts for label-efficient object detection and tracking experiments on high-resolution drone imagery. The workflow compares fully supervised baselines against self-supervised pretraining pipelines to reduce the amount of labeled data required for downstream detection.

The included notebooks cover data exploration, supervised detector training, self-supervised representation learning, model transfer/fine-tuning, comparative evaluation, error analysis, and tracking.

## Problem Statement
High-quality annotations for military aerial imagery are expensive and time-consuming to produce. This project investigates how self-supervised learning (SSL) and transfer strategies can improve detector and tracker performance when only a subset of labels is available.

## Expected Use Cases
- Rapid experimentation for label-efficient military aerial object detection.
- Comparing supervised vs SSL-initialized detector performance.
- Reproducing classroom/research notebook pipelines for benchmarking.
- Generating evaluation and error-analysis outputs for model selection.
- Building a detection-to-tracking workflow on drone footage.

## Features and Capabilities
- **Dataset preparation and EDA** notebooks for split checks and annotation integrity.
- **Supervised detection baselines** (YOLO-family and RF-DETR notebooks).
- **Self-supervised pretraining pipelines** (e.g., SimCLR, BYOL, I-JEPA, DINOv3-oriented workflows).
- **Transfer and detection fine-tuning** notebooks for low-label settings.
- **Comparative evaluation and error analysis** notebooks.
- **Tracking pipeline notebook** for integrating detection and tracking.

> Note: This repository is notebook-first. Some paths, dataset locations, and runtime configuration values are expected to be set in notebook config cells.

## Repository Structure
```text
.
├── LICENSE
├── README.md
├── skills.md
├── notebook-1-group-b.ipynb                       # Dataset exploration & preprocessing
├── notebook-2-yolo-model-training-yolov10-v12-v26.ipynb
├── notebook-3-rf-detr-nano-training.ipynb
├── notebook-4-evaluation-error-analysis.ipynb
├── notebook-5-rf-detr.ipynb
├── group02-partb-1a-simclr-pretrain.ipynb
├── group02-partb-1b-simclr-detect.ipynb
├── group02-partb-02a-byol-pretrain.ipynb
├── group02-partb-02b-byol-detect.ipynb
├── group02-partb-03a-ijepa-pretrain.ipynb
├── group02-partb-03b-ijepa-detect.ipynb
├── group02-partb-04a-dinov3-pretrain.ipynb
├── group02-partb-04b-dinov3-detect.ipynb
├── group02-partb-05-tracking.ipynb
└── group02-partb-bonus-label-efficiency.ipynb
```

## Setup
Because this repository is notebook-centric, install dependencies in your notebook environment (local Jupyter or Google Colab).

### Prerequisites
- Python 3.10+
- Jupyter Notebook / JupyterLab or Google Colab
- (Recommended) GPU runtime for training and SSL pretraining

### Suggested Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter numpy pandas matplotlib seaborn pillow opencv-python pyyaml torch torchvision
```

### Notebook-Specific Dependencies
Depending on which notebook you run, you may also need:
- `ultralytics`
- `rfdetr`
- `supervision`
- `pycocotools`
- `ssldet`
- `lightly-train`

Install them in the corresponding notebook/setup cell as required.

## Usage and Workflow
A practical execution order is:

1. **Dataset exploration and preprocessing**  
   Run `notebook-1-group-b.ipynb` to validate data and prepare splits.

2. **Supervised detector baselines**  
   Run `notebook-2-yolo-model-training-yolov10-v12-v26.ipynb` and/or `notebook-3-rf-detr-nano-training.ipynb`.

3. **Self-supervised pretraining + transfer**  
   Use the paired pretrain/detect notebooks:
   - SimCLR: `group02-partb-1a-simclr-pretrain.ipynb` → `group02-partb-1b-simclr-detect.ipynb`
   - BYOL: `group02-partb-02a-byol-pretrain.ipynb` → `group02-partb-02b-byol-detect.ipynb`
   - I-JEPA: `group02-partb-03a-ijepa-pretrain.ipynb` → `group02-partb-03b-ijepa-detect.ipynb`
   - DINOv3-oriented: `group02-partb-04a-dinov3-pretrain.ipynb` → `group02-partb-04b-dinov3-detect.ipynb`

4. **Tracking and winner analysis**  
   Run `group02-partb-05-tracking.ipynb` for comparison and tracking integration.

5. **Evaluation and error analysis**  
   Run `notebook-4-evaluation-error-analysis.ipynb` (and related evaluation sections) to compare models.

### Inference (Generic)
For inference in your selected notebook:
1. Load the best checkpoint path in the notebook config cell.
2. Set inference source (image directory/video stream).
3. Run prediction and visualization/export cells.

## Evaluation and Metrics
Use the evaluation notebooks to report task-relevant metrics. Typical metrics include:

- **Detection**: mAP@0.5, mAP@0.5:0.95, precision, recall
- **Tracking**: MOTA, IDF1, HOTA, ID switch count (if computed)
- **Efficiency**: FPS/latency and memory usage (optional)

You can maintain a summary table like:

| Model / Pipeline | Label Fraction | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---:|---|
| YOLO Baseline | TODO | TODO | TODO | TODO | TODO | |
| RF-DETR Nano | TODO | TODO | TODO | TODO | TODO | |
| SimCLR + Detector | TODO | TODO | TODO | TODO | TODO | |
| BYOL + Detector | TODO | TODO | TODO | TODO | TODO | |
| I-JEPA + Detector | TODO | TODO | TODO | TODO | TODO | |
| DINOv3 + Detector | TODO | TODO | TODO | TODO | TODO | |

## Suggested GitHub Repository Description
Use this concise repository description in GitHub settings:

**Label-efficient object detection and tracking in military drone imagery, comparing supervised and self-supervised pipelines for low-annotation scenarios.**

(Also saved in `REPOSITORY_DESCRIPTION.txt`.)

## Contributing
Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes with clear notebook or documentation updates.
4. Open a pull request describing:
   - what changed,
   - why it changed,
   - how to reproduce/validate.

Please keep changes modular and avoid committing large generated artifacts unless required for reproducibility.

## License
This project is licensed under the **Apache License 2.0**. See [`LICENSE`](./LICENSE).
