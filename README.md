# Visually Mapping the Algorithmic Breaking Point: Grad-CAM Explainability of Focal Loss Failure Under Extreme Data Starvation in Sri Lankan Tea Leaf Disease Detection

[![Target Journal: IEEE Access](https://img.shields.io/badge/Target%20Journal-IEEE%20Access-blue.svg)](https://ieeeaccess.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![TensorFlow: 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

---

## 📌 Project Overview
**Authors:**
- **E. Dasun Manjitha** (Student ID: `ITBIN-2313-0062`)
- **A. G. S. Lakshan** (Student ID: `ITBIN-2313-0055`)

**Institution:** Horizon Campus, Faculty of IT  
**Module:** IT41043 - Intelligent Systems  
**Module Coordinator:** Mr. Isuru Madusanka Samarappulige  
**Submission Date:** October 2026  
**Target Publication Venue:** *IEEE Access* (Rapid 4–6 week review cycle, Gold Open Access with developing country APC waiver eligibility)

---

## 🔬 Section A: The Gap Statement
Tea (*Camellia sinensis*) is a foundational pillar of the Sri Lankan agricultural export economy. Foliar pathogens cause massive annual crop yields and economic losses if not identified early. While modern Convolutional Neural Networks (CNNs) have shown state-of-the-art diagnostic performance under laboratory benchmarks, real-world deployment faces severe distributional anomalies:

1. **Unrealistic Data Balance Assumptions:** Existing agricultural computer vision literature presumes access to well-balanced or mildly imbalanced disease datasets (e.g., natural 10:1 ratios). In real operational tea plantations in Sri Lanka, healthy leaves are overwhelmingly abundant, whereas sporadic endemic disease outbreaks create extreme minority class representation.
2. **Untested Algorithmic Breaking Points:** Dynamically weighted classification losses such as **Focal Loss** are widely adopted to handle class imbalance by down-weighting well-classified examples. However, current literature relies predominantly on macro-averaged metrics on mild imbalances, failing to establish the precise mathematical ratio where Focal Loss collapses.
3. **The Misuse of Explainable AI (XAI):** Gradient-weighted Class Activation Mapping (Grad-CAM) is traditionally used to highlight model success. In this project, we employ Grad-CAM to visually diagnose **algorithmic collapse**, proving that under extreme data starvation, networks fail by overfitting to incidental background artifacts and leaf texture noise rather than true pathological lesions.

---

## ❓ Section B: Core Research Question
> **"At what specific extreme data starvation ratio (e.g., 1:100, 1:500) does the Focal Loss function mathematically fail to protect minority-class Sri Lankan tea leaf diseases, and how does this algorithmic collapse visually manifest in Grad-CAM spatial attention heatmaps?"**

---

## 🎯 Section C: Scope Definition
- **In Scope:**
  - Curation of local Sri Lankan tea leaf disease image sets across 7 classes: *Tea algal leaf spot*, *Brown Blight*, *Gray Blight*, *Helopeltis*, *Red spider*, *Green mirid bug*, and *Healthy leaf*.
  - Inducing controlled synthetic class starvation to simulate real-world minority disease rarity: **Baseline** (natural distribution), **Severe (1:100)**, and **Extreme (1:500)**.
  - Transfer learning benchmarks using **DenseNet121** feature extraction comparing **Standard Categorical Cross-Entropy** against **Focal Loss** ($\gamma=2.0, \alpha=0.25$).
  - Rigorous statistical evaluation using Macro/Minority-class Precision, Recall, and F1-Scores validated with 5-fold cross-validation.
  - Spatial attention explainability with **Grad-CAM** (`tf-keras-vis`) to map attention scattering and background hallucination.
- **Out of Scope:**
  - Real-time embedded edge-device field deployments.
  - Multimodal sensory integration (e.g., weather telemetry, soil moisture).
  - Generative AI synthetic oversampling (e.g., diffusion models or GANs).

---

## 🛠️ Repository Structure
```
├── Merged_Tea_Leaf_Dataset/  # Master local raw tea leaf image dataset (Ignored by Git)
├── Baseline/                 # Natural imbalance benchmark dataset
├── Severe_1_100/             # Induced 1:100 starvation dataset
├── Extreme_1_500/            # Induced 1:500 starvation dataset
├── data_starvation.py        # Automated starvation engineering script
├── train_dense_net.py        # DenseNet121 training with Cross-Entropy & Focal Loss
├── gradcam_explainability.py # Grad-CAM attention visualizer
├── requirements.txt          # Python dependencies
├── .gitignore                # Version control ignore rules
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Ensure Python 3.10+ is installed, then set up the environment:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation & Starvation
Execute the data engineering script to generate the synthetic starvation regimes:
```bash
python data_starvation.py
```

### 3. Training & Stress Testing
Run the comparative training pipeline:
```bash
python train_dense_net.py
```

### 4. Grad-CAM Visualization
Generate comparative spatial attention heatmaps:
```bash
python gradcam_explainability.py
```
