
# Label-Efficient Object Detection and Tracking in Military Drone Imagery: A Comparative Experimental Study of Supervised Detectors and Self-Supervised Backbones

CSE 445 (Computer Vision), Group 02 — Supervisor: Dr. Mohammad Rifat Ahmmad Rashid

Authors:
- Md. Ariful Islam Opi – 2023-1-60-141
- MD Shafin Ahamed Siam – 2023-1-60-007
- Albin Israt Siddique – 2023-1-60-212
- Nasrullah Kaisher Sijan – 2023-1-60-204

Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh

Abstract

Object detection in military drone imagery is constrained by the high cost of expert annotation. This project studies two complementary questions: (A) which supervised detector architecture performs best under a fixed compute budget, and (B) whether self-supervised pretraining of detector backbones improves label efficiency when fine-tuned on limited labelled data. We compare four supervised detectors and four self-supervised learning (SSL) pretraining methods on a top-down military aerial dataset (KIIT-MiTA).

Index Terms: object detection, self-supervised learning, label efficiency, multi-object tracking, YOLO, RF-DETR, SimCLR, BYOL, I-JEPA, DINOv3, military imagery

## Contents

- Introduction
- Methods (Part A: Supervised detectors; Part B: SSL pretraining)
- Weight surgery and fine-tuning
- Tracking pipeline
- Results
- Figures and captions (placeholders)
- Limitations and future work
- Conclusion
- References
- Contribution

## I. Introduction

Detecting military hardware (tanks, artillery, radar installations, missile launchers, soldiers, and support vehicles) in top-down drone imagery is a challenging, low-resource problem: imagery is often restricted, objects are rare, and correct labelling requires domain expertise. We evaluate detector architectures and SSL pretraining methods under a constrained compute budget (Kaggle, 2 × NVIDIA T4) and limited labelled data.

## II. High-level overview

Part A — Model selection: compare four supervised detector architectures trained on the same training/validation/test partition under comparable budgets.

Part B — Label-efficiency: pretrain YOLOv26n backbones using four SSL methods on an unlabelled pool, then transfer the pretrained encoders into a YOLOv26n detector and fine-tune on fractions of the labelled data.

A dataset-specific augmentation (copy-paste) was used to reduce class imbalance in the SSL pool (see Table II).

## Table II: Effect of copy-paste augmentation on SSL pool class balance (train-split)

| Class | Before | After |
|---|---:|---:|
| Artillery (minority) | 240 | 610 |
| Radar (minority) | 225 | 772 |
| M. Rocket Launcher (minority) | 311 | 705 |
| Missile | 362 | 764 |
| Tank | 643 | 824 |
| Soldier | 907 | 937 |
| Vehicle (majority) | 881 | 1,033 |
| Imbalance ratio (max:min) | 4.03× | 1.69× |
| Train images | 1,339 | 2,377 |

## Methods (summary)

Part A — Supervised detectors evaluated:
- YOLOv10-s
- YOLOv12-s
- YOLOv26-s
- RF-DETR-nano

Part B — SSL pretraining methods:
- SimCLR
- BYOL
- I-JEPA
- DINOv3-guided distillation

All SSL methods pretrained a YOLOv26n backbone (or were distilled into one for DINOv3) on the 1,339-image SSL pool at 224×224 resolution, batch size 32/GPU (64 global), for 200 epochs on 2×T4 GPUs.

### Weight surgery and fine-tuning

Pretrained encoder weights were loaded into corresponding detector backbone layers ("weight surgery") and verified for compatibility before fine-tuning. Downstream fine-tuning at 20% labelled data used a two-stage protocol: a short head warm-up followed by joint training of all detector parameters.

### Tracking pipeline

The best-performing SSL-initialised detector was deployed in a tracking-by-detection pipeline using ByteTrack for association.

## Results (selected tables)

### Table III: Part A — consolidated detector comparison (test split)

| Model | mAP50 | mAP50-95 | Prec. | Rec. | F1 | FPS | Params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv10-s | 0.646 | 0.421 | 0.673 | 0.600 | 0.635 | 84.7 | 7.22 |
| YOLOv12-s | 0.665 | 0.432 | 0.631 | 0.636 | 0.633 | 71.2 | 9.23 |
| **YOLOv26-s** | 0.648 | **0.460** | 0.722 | 0.624 | 0.669 | 82.1 | 9.47 |
| RF-DETR-n | **0.754** | **0.526** | 0.724 | 0.706 | 0.715 | 20.6 | 30.45 |

Key findings (brief):
- RF-DETR-nano achieved the highest overall accuracy (mAP50-95 = 0.526) but is architecturally incompatible with the SSL weight-surgery pipeline used in Part B.
- Among SSL methods at 20% labels, SimCLR produced the strongest downstream detector (mAP50-95 = 0.0642), but no SSL method matched a COCO-pretrained baseline across the tested label fractions up to 50%.


## Limitations and future work

- Compute-constrained pretraining (200 epochs on 2×T4) may be insufficient for methods like DINOv3 distillation to fully converge.  
- Results are specific to the KIIT-MiTA dataset (1,681 images) and the top-down aerial domain; generalization to other domains is not claimed.  
- One tracking clip is AI-generated; synthetic video may differ from real footage in texture, motion, and lighting.

## Conclusion

SimCLR gave the best SSL backbone transfer in our experiments, but SSL pretraining on the tested scale and budget did not outperform COCO-pretrained supervised initialization for this dataset and label regimes.

## References

- [1] KIIT-MiTA Dataset, Mendeley Data. Available: https://data.mendeley.com/datasets/drjmrf5kk5/1  
- [2] M. R. A. Rashid, “ssl-detection-lab,” 2024. GitHub repository: https://github.com/rifat963/ssl-detection-lab  
- [3] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLO,” 2023. GitHub: https://github.com/ultralytics/ultralytics  
- [4] A. Wang et al., “YOLOv10: Real-time end-to-end object detection,” NeurIPS, 2024.  
- [5] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attention-centric realtime object detectors,” arXiv:2502.12524, 2025.  
- [6] Roboflow, “RF-DETR: A real-time, DETR-based object detection model,” 2025. GitHub: https://github.com/roboflow/rf-detr  
- [7] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations (SimCLR),” Proc. ICML, 2020.  
- [8] J.-B. Grill et al., “Bootstrap Your Own Latent (BYOL),” 2020.  
- [9] M. Assran et al., “Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA),” Proc. (conference), year.  
- [10] O. Siméoni et al., “DINOv3,” Meta AI Research, 2024.  
- [11] Y. Zhang et al., “ByteTrack: Multi-object tracking by associating every detection box,” Proc. ECCV, 2022.  
- [12] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, “BoT-SORT: Robust associations multi-pedestrian tracking,” arXiv:2206.14651, 2022.

(If you want full BibTeX entries I can expand these citations into full reference formats.)

## Contribution

| Member | Notebooks / Report Sections Led |
|---|---|
| Md. Ariful Islam Opi (2023-1-60-141) | EDA Notebook; SimCLR notebooks; Tracking notebook; Part A RF-DETR; Report Sections III-B1, V-C |
| MD Shafin Ahamed Siam (2023-1-60-007) | BYOL notebooks; Part A YOLOv10; Report Section III-B2 |
| Albin Israt Siddique (2023-1-60-212) | I-JEPA notebooks; Part A YOLOv12; Report Section III-B3 |
| Nasrullah Kaisher Sijan (2023-1-60-204) | DINOv3 notebooks; Bonus notebook; Part A YOLOv26; Report Section III-B4 |
