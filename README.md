# Visually Mapping the Algorithmic Breaking Point: Grad-CAM Explainability of Focal Loss Failure Under Extreme Data Starvation in Sri Lankan Tea Leaf Disease Detection

[![Target Journal: IEEE Access](https://img.shields.io/badge/Target%20Journal-IEEE%20Access-blue.svg)](https://ieeeaccess.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![TensorFlow: 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Milestone 2: Complete](https://img.shields.io/badge/Milestone%202-100%25%20Verified-brightgreen.svg)]()

---

## 📌 Project Overview
- **Institution:** Horizon Campus, Faculty of Information Technology  
- **Degree Program:** BSc (Hons) in Information Technology / Intelligent Systems  
- **Module:** IT41043 — Intelligent Systems (Academic Year 2026, 3rd Year, 2nd Semester)  
- **Module Coordinator:** Mr. Isuru Madusanka Samarappulige  
- **Target Journal:** *IEEE Access* (Rapid 4–6 week review cycle, Gold Open Access with developing country APC waiver eligibility)  

---

## 👥 Research Team & Contributions
| Student Researcher | Student ID | Primary Technical Domain & Repository Contributions |
| :--- | :---: | :--- |
| **E. Dasun Manjitha** | `ITBIN-2313-0062` | Master dataset curation, automated synthetic data starvation engineering (`data_starvation.py`), Focal Loss stress-testing optimization (`train_dense_net.py`), and mathematical evaluation metric formulation. |
| **A. G. S. Lakshan** | `ITBIN-2313-0055` | DenseNet121 transfer learning backbone implementation, Grad-CAM spatial explainability pipeline (`gradcam_explainability.py`), vector architecture visualization, and interactive testing suite (`test_prediction.py`). |

---

## 📅 Project State and Milestones

| Milestone | Deliverable Description | Status | Verification & Deliverables |
| :---: | :--- | :---: | :--- |
| **Milestone 1** | Research Proposal, Gap Statement, Scope, Annotated Bibliography | **Completed** | Proposal approved with primary focus on Sri Lankan tea pathology. |
| **Milestone 2** | Methodology & Data Description (Dataset, Architecture Diagram, Baseline, Evaluation Plan, Repo, Bibliography) | **Current / 100% Complete** | Document in `milestone2_deliverables/Milestone_2_Document.md` & public GitHub repo. |
| **Milestone 3** | Mid-Term Progress Checkpoint & Model Convergence Verification | **In Progress** | DenseNet121 trained across Baseline, 1:100, and 1:500 regimes. |
| **Milestone 4** | Complete Academic Research Paper Submission (Assignment 4) | **Draft Ready** | 3,000+ word academic draft prepared targeting IEEE Access. |
| **Milestone 5** | Proof of Research Paper Submission to Conference/Journal (Assignment 5) | **Scheduled** | Manuscript submission receipt and tracking verification. |

---

## 🔬 Research Motivation and Objectives

### 1. Agronomic & Economic Context
Tea (*Camellia sinensis*) is a foundational pillar of the Sri Lankan export economy. Foliar pathogens—including *Algal Leaf Spot*, *Brown Blight*, *Gray Blight*, *Helopeltis*, *Red Spider Mite*, and *Green Mirid Bug*—induce severe tissue necrosis and crop yield losses if not diagnosed early. While modern Convolutional Neural Networks (CNNs) show stellar diagnostic accuracy on balanced benchmarks, real commercial tea plantations present extreme, sporadic data starvation where healthy foliage vastly outnumbers localized diseased leaves by ratios exceeding 1:100 to 1:500.

### 2. Core Research Gap
1. **Unrealistic Balance Assumptions:** Prior agricultural vision works evaluate models on artificial 1:1 or mild 10:1 imbalances.
2. **Untested Failure Boundaries:** While dynamically weighted classification losses such as **Focal Loss** ($\gamma=2.0, \alpha=0.25$) theoretically down-weight easy healthy examples, no research has artificially starved models to locate the **exact mathematical ratio where Focal Loss collapses**.
3. **Forensic Use of Explainable AI (XAI):** Rather than using Grad-CAM to celebrate AI success, we employ it as a diagnostic instrument to visually prove that under extreme data starvation, networks fail by overfitting to **incidental background leaf margins and specular lighting noise** rather than true pathological lesions.

### 3. Core Research Question
> **"At what specific extreme data starvation ratio (e.g., 1:100, 1:500) does the Focal Loss function mathematically fail to protect minority-class Sri Lankan tea leaf diseases, and how does this algorithmic collapse visually manifest in Grad-CAM spatial attention heatmaps?"**

---

## 🏛️ System Architecture Pipeline

The complete end-to-end research methodology is visually modeled below:

```mermaid
flowchart TD
    classDef inputStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef prepStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef modelStyle fill:#312e81,stroke:#c084fc,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef branch1Style fill:#450a0a,stroke:#f87171,stroke-width:2px,color:#fef2f2;
    classDef branch2Style fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5;
    classDef outStyle fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#eff6ff;

    subgraph Stage1 ["Stage 1: Multi-Regime Data Ingestion"]
        D1["Baseline Dataset<br/>(1:1 Natural Distribution: 1000 H / 995 D)"]:::inputStyle
        D2["Severe Starvation Dataset<br/>(1:100 Ratio: 1000 H / 10 D)"]:::inputStyle
        D3["Extreme Starvation Dataset<br/>(1:500 Ratio: 1000 H / 2 D)"]:::inputStyle
    end

    subgraph Stage2 ["Stage 2: Deterministic Preprocessing Pipeline"]
        P1["Bilinear Spatial Resizing: 224 x 224 x 3"]:::prepStyle
        P2["Pixel Intensity Normalization: [0.0, 1.0]"]:::prepStyle
        P3["Stratified 5-Fold Cross-Validation Partitioning"]:::prepStyle
        D1 --> P1
        D2 --> P1
        D3 --> P1
        P1 --> P2 --> P3
    end

    subgraph Stage3 ["Stage 3: DenseNet121 Feature Reuse Extractor"]
        M1["Pre-trained DenseNet121 Convolutional Backbone<br/>(4 Dense Blocks with Direct Layer Concatenation)"]:::modelStyle
        M2["Global Average Pooling 2D (1024-dim Vector)"]:::modelStyle
        M3["Dense Softmax Classification Head (C = 2 Classes)"]:::modelStyle
        P3 --> M1 --> M2 --> M3
    end

    subgraph Stage4 ["Stage 4: Branching Optimization Regimes"]
        B1["Branch A: Categorical Cross-Entropy (Baseline)<br/>L_CE = -&Sigma; y_i log(p_i)<br/>&bull; Vulnerable to Majority Gradient Swamping"]:::branch1Style
        B2["Branch B: Focal Loss Stress Test (&gamma;=2.0, &alpha;=0.25)<br/>L_FL = -&alpha;_t (1 - p_t)^&gamma; log(p_t)<br/>&bull; Downweights Easy Healthy Leaves"]:::branch2Style
        M3 --> B1
        M3 --> B2
    end

    subgraph Stage5 ["Stage 5: Dual Evaluation & Visual Explainability"]
        O1["Quantitative Metrics Suite<br/>&bull; Precision, Recall, Minority F1, Macro F1<br/>&bull; Paired t-tests & Wilcoxon Signed-Rank Tests"]:::outStyle
        O2["Qualitative XAI (tf-keras-vis Grad-CAM)<br/>&bull; Target Layer: conv5_block16_concat<br/>&bull; Lesion Localization vs Attention Scattering"]:::outStyle
        B1 --> O1
        B2 --> O1
        B1 --> O2
        B2 --> O2
    end
```

---

## ⚙️ END-TO-END PIPELINE STAGES

### Stage 1: Controlled Data Starvation Ingestion
The master local repository contains **10,143 high-resolution images** across 7 classes. To isolate the algorithmic collapse threshold:
- **Baseline (1:1):** 1,000 Healthy vs. 995 Diseased leaves.
- **Severe (1:100):** 1,000 Healthy vs. 10 Diseased leaves (1 diseased specimen per 100 healthy leaves).
- **Extreme (1:500):** 1,000 Healthy vs. 2 Diseased leaves (1 diseased specimen per 500 healthy leaves).
- **Test Set:** 200 unseen images (100 Healthy, 100 Diseased) reserved strictly for unbiased post-training evaluation.

### Stage 2: Data Preprocessing Pipeline
1. **Resizing:** Bilinear interpolation to $224 \times 224 \times 3$.
2. **Normalization:** Scaling 8-bit integer pixels $[0, 255]$ by $\frac{1}{255.0}$ to continuous floating-point $[0.0, 1.0]$.
3. **Data Augmentation:** Random horizontal reflections and slight rotations ($\pm 15^\circ$) to simulate varied in-situ field angles without corrupting class labels.

### Stage 3: Deep Feature Extraction Backbone
DenseNet121 connects every layer to all subsequent layers:
$$x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])$$
This design maximizes feature reuse, mitigates the vanishing gradient problem, and operates efficiently with only ~7.03M parameters. Feature maps from `conv5_block16_concat` pass through GlobalAveragePooling2D, Dropout ($p=0.2$), and a two-neuron Dense Softmax output layer.

### Stage 4: Branching Optimization & Stress Testing
- **Branch A (Categorical Cross-Entropy):** Standard unweighted loss prone to gradient swamping under extreme majority imbalance.
- **Branch B (Focal Loss):** Introduces dynamic modulating factor $(1 - p_t)^\gamma$ with $\gamma = 2.0$ and $\alpha = 0.25$ to heavily down-weight easy majority samples.

### Stage 5: Evaluation & Visual Explainability
- **Mathematical Evaluation:** Precision, Recall, Minority $F_1$, and Macro $F_1$ evaluated across 5-Fold Stratified Cross-Validation.
- **Grad-CAM Attention Mapping:** Visualizing gradient flow at `conv5_block16_concat` using `tf-keras-vis`.

---

## 📊 Model Benchmarking & Stress-Testing Results

The pipeline was executed across all six configurations on the independent balanced test partition (200 unseen images):

| Dataset Regime | Starvation Ratio | Loss Function | Accuracy (%) | Minority Precision | Minority Recall | Minority F1-Score | Macro F1-Score | Algorithmic State |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 1:1 | Categorical CE | 92.00% | 0.9667 | 0.8700 | **0.9158** | **0.9198** | Robust Lesion Detection |
| **Baseline** | 1:1 | Focal Loss ($\gamma=2$) | 93.00% | 0.9778 | 0.8800 | **0.9263** | **0.9298** | Robust Lesion Detection |
| **Severe** | 1:100 | Categorical CE | 51.50% | 1.0000 | 0.0300 | **0.0583** | **0.3658** | Majority Class Swamping |
| **Severe** | 1:100 | Focal Loss ($\gamma=2$) | 50.50% | 1.0000 | 0.0100 | **0.0198** | **0.3443** | Degraded / Breakdown Phase |
| **Extreme** | 1:500 | Categorical CE | 50.00% | 0.0000 | 0.0000 | **0.0000** | **0.3333** | **Catastrophic Model Collapse** |
| **Extreme** | 1:500 | Focal Loss ($\gamma=2$) | 50.00% | 0.0000 | 0.0000 | **0.0000** | **0.3333** | **Catastrophic Model Collapse** |

> **Mathematical Proof of Failure:** At the 1:500 ratio, the volume of 500 healthy leaves completely overwhelms the gradient contribution of the single diseased leaf, driving Minority Recall and F1 to **0.0000**. This mathematically locates the failure breaking point between 1:100 and 1:500.

---

## 🔍 Explainable AI (Grad-CAM) Visual Proof

Using `tf-keras-vis`, we visually map how the network's internal attention breaks down under extreme starvation:

- **Baseline Model (Normal Data):** Focuses high-intensity heat (crimson red / yellow) directly onto the pathogen lesion on the tea leaf blade. Surrounding healthy green tissue produces zero activation.
- **Extreme 1:500 Model (Starved):** Exhibits complete **attention scattering**. The actual lesion is ignored ($0\%$ activation), while the network hallucinates focus onto irrelevant background leaf margins and specular lighting artifacts.

*(View the high-resolution comparison graphic locally in `results/gradcam_comparison.png` or inspect `milestone2_deliverables/architecture_diagram.svg`)*

---

## 🛠️ Repository Structure
```
├── milestone2_deliverables/        # Milestone 2 submission package
│   ├── Milestone_2_Document.md    # Complete 6-section brief for single PDF submission
│   ├── architecture_diagram.svg    # Standalone high-resolution vector SVG flowchart
│   ├── architecture_diagram.md     # Mermaid vector flowchart
│   ├── mathematical_evaluation_plan.md # Metric derivations & 5-fold CV protocol
│   └── README.md                   # Overview of milestone deliverables
├── data_starvation.py              # Automated data starvation engineering script
├── train_dense_net.py              # DenseNet121 training under Cross-Entropy vs Focal Loss
├── gradcam_explainability.py       # Grad-CAM spatial attention visualizer (tf-keras-vis)
├── test_prediction.py              # Interactive single-image model prediction test script
├── requirements.txt                # Python environment dependencies
├── .gitignore                      # Version control ignore rules (data/weights/drafts)
└── README.md                       # Master repository documentation
```

---

## 🚀 Getting Started & Reproducibility Guide

### 1. Environment Setup
```bash
git clone https://github.com/E-Dasun-Manjitha/Tea-Leaf-Focal-Loss-Research.git
cd Tea-Leaf-Focal-Loss-Research
pip install -r requirements.txt
```

### 2. Run Data Starvation Preprocessing
Generates Baseline, Severe (1:100), Extreme (1:500), and test partitions:
```bash
python data_starvation.py
```

### 3. Execute Model Training & Stress Testing
Trains DenseNet121 under both CCE and Focal Loss across all 3 regimes:
```bash
python train_dense_net.py
```

### 4. Generate Grad-CAM Heatmaps
Produces side-by-side attention scattering visualizations:
```bash
python gradcam_explainability.py
```

### 5. Interactive Single-Image Model Testing
Test any tea leaf image to witness algorithmic plant blindness:
```bash
python test_prediction.py "Test_Set/Diseased/20211205_160445.jpg"
```

---

## 📚 Revised Bibliography (15 Peer-Reviewed Sources)

1. **[IEEE Xplore]** X. Deng and C. Photong, "Evaluation of Tea Leaf Disease Identification Based on Convolutional Neural Networks VGG16, ResNet50, and DenseNet169 Image Recognitions," in *2024 12th iEECON*, 2024.
2. **[IEEE Xplore]** M. S. H. Shovon et al., "PlantDet: A Robust Multi-Model Ensemble Method Based on Deep Learning For Plant Disease Detection," *IEEE Access*, vol. 11, 2023.
3. **[IEEE Xplore]** A. Oad et al., "Plant Leaf Disease Detection Using Ensemble Learning and Explainable AI," *IEEE Access*, vol. 12, 2024.
4. **[IEEE Xplore]** M. Raj et al., "CNN and Random Forest Hybrid Model for Tea Leaf Diseases Multiclassification," in *2024 AUTOCOM*, 2024.
5. **[IEEE Xplore]** T.-Y. Lin et al., "Focal Loss for Dense Object Detection," *IEEE TPAMI*, vol. 42, no. 2, 2020.
6. **[ACM DL]** D. Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection," in *Proc. 7th ACM IKDD CoDS / 25th COMAD*, 2020.
7. **[ACM DL]** D. Zhai et al., "Rectified Meta-learning from Noisy Labels for Robust Image-based Plant Disease Classification," *ACM TOMM*, vol. 18, 2022.
8. **[ACM DL]** G. Fenu and F. M. Malloci, "An Application of Machine Learning Technique in Forecasting Crop Disease," in *Proc. 2019 3rd ACM Big Data Res.*, 2019.
9. **[Springer]** A. Chug et al., "A novel framework for image-based plant disease detection using hybrid deep learning approach," *Soft Computing*, vol. 27, 2023.
10. **[Springer]** P. Bhuyan et al., "Res4net-CBAM: a deep cnn with convolution block attention module for tea leaf disease diagnosis," *Multimed Tools Appl*, vol. 83, 2023.
11. **[Springer]** S. Mukhopadhyay et al., "Tea leaf disease detection using deep learning with transfer learning approaches," *Applied Intelligence*, vol. 52, 2022.
12. **[Springer]** H. Wang et al., "Identification of tea leaf diseases based on dual-pathway convolutional neural network with channel attention," *J Plant Pathol*, vol. 105, 2023.
13. **[Nature / Sci Rep]** Md. J. A. Soeb et al., "Tea leaf disease detection and identification based on YOLOv7 (YOLO-T)," *Scientific Reports*, vol. 13, 2023.
14. **[PLoS One]** J. Yang et al., "Lightweight wavelet-CNN tea leaf disease detection," *PLoS One*, vol. 20, 2025.
15. **[Hindawi / Wiley]** J. A. Pandian et al., "Grey Blight Disease Detection on Tea Leaves Using Improved Deep Convolutional Neural Network," *Comput Intell Neurosci*, 2023.
