# ECG Beat Classification with 1D CNN and Focal Loss

This repository contains a complete PyTorch-based pipeline for classifying ECG heartbeats using data from the MIT-BIH Arrhythmia Database. The goal is to classify each heartbeat into one of five broad categories using a 1D Convolutional Neural Network (CNN) architecture, with enhancements like focal loss and one attention block.

---

## Overview

This project performs beat-level classification of ECG signals extracted from the MLII lead of the MIT-BIH Arrhythmia dataset. R-peak-centered segments of the ECG signals are extracted and passed through a CNN model designed for efficient and robust heartbeat classification. The model addresses significant class imbalance using a custom Focal Loss implementation and explores temporal attention as an enhancement.

---

## Results

| Metric        | Value    |
|---------------|----------|
| Accuracy      | 98.70%   |
| Macro F1      | 0.9312   |
| Macro Precision | 0.9326 |
| Best per-class F1 | ~0.99 (Normal), ~0.97 (Ventricular), ~0.88 (Supraventricular) |

---

## Dataset

- **Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Sampling Rate:** 360 Hz
- **Used Lead:** MLII
- **Beat Segments:** 300-sample windows centered around annotated R-peaks
- **Labels Mapped To:**
  - Class 0: Normal (N, L, R, etc.)
  - Class 1: Ventricular ectopic beats (V, E)
  - Class 2: Supraventricular ectopic beats (A, a, J, S)
  - Class 3: Fusion (F)
  - Class 4: Unknown ('/', 'f', 'Q', '|')

---

## References

- Moody GB, Mark RG. "The MIT-BIH Arrhythmia Database." PhysioNet.
- Lin et al., "Focal Loss for Dense Object Detection", [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- WFDB Python Toolkit: https://github.com/MIT-LCP/wfdb-python
- IET Review of ECG Denoising: [IET Signal Processing](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-spr.2020.0104)
