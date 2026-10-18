# System Architecture Pipeline

The complete end-to-end research methodology is visually modeled below:

```mermaid
flowchart TD
    subgraph S1 ["Stage 1: Multi-Regime Data Ingestion"]
        D1["Baseline Dataset (1:1 Natural Ratio)
        1,000 Healthy / 995 Diseased"]
        D2["Severe Starvation Dataset (1:100 Ratio)
        1,000 Healthy / 10 Diseased"]
        D3["Extreme Starvation Dataset (1:500 Ratio)
        1,000 Healthy / 2 Diseased"]
    end

    subgraph S2 ["Stage 2: Deterministic Preprocessing & Partitioning"]
        P1["Bilinear Spatial Resizing: 224 x 224 x 3"]
        P2["Pixel Intensity Normalization: [0.0, 1.0]
        Formula: pixel_norm = pixel / 255.0"]
        P3["Dual Partitioning Strategy:
        • Stratified 5-Fold Cross-Validation (Training / Val)
        • Independent Test Holdout (100 Healthy + 100 Diseased)"]
    end

    subgraph S3 ["Stage 3: DenseNet121 Feature Reuse Extractor"]
        M1["Pre-trained DenseNet121 Backbone
        (4 Dense Blocks with Direct Layer Concatenation)"]
        M2["Global Average Pooling 2D
        (1024-dimensional Feature Vector)"]
        M2B["Dropout Regularization
        (Rate p = 0.2)"]
        M3["Dense Softmax Classification Head
        (C = 2 Diagnostic Classes)"]
    end

    subgraph S4 ["Stage 4: Branching Optimization Regimes"]
        B1["Branch A: Categorical Cross-Entropy (Baseline)
        L_CE = -sum( y_i * log(p_i) )
        • Vulnerable to Majority Gradient Swamping"]
        B2["Branch B: Focal Loss Stress Test (gamma=2.0, alpha=0.25)
        L_FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        • Dynamically Downweights Easy Healthy Leaves"]
    end

    subgraph S5 ["Stage 5: Dual Evaluation & Forensic Explainability"]
        O1["Quantitative Statistical Suite
        • Precision, Recall, Minority F1, Macro F1
        • Stratified 5-Fold CV Variance & Confusion Matrix
        • Paired t-tests & Wilcoxon Signed-Rank Tests (p < 0.05)"]
        O2["Qualitative Forensic XAI (tf-keras-vis Grad-CAM)
        • Target Feature Layer: conv5_block16_concat
        • True Lesion Localization vs Specular Noise Overfitting"]
    end

    D1 --> P1
    D2 --> P1
    D3 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> M1
    M1 --> M2
    M2 --> M2B
    M2B --> M3
    M3 --> B1
    M3 --> B2
    B1 --> O1
    B2 --> O1
    B1 --> O2
    B2 --> O2
```

---

### Architectural Flow Summary
1. **Multi-Regime Ingestion:** Models three controlled environments (Natural Baseline 1:1, Severe Starvation 1:100, and Extreme Starvation 1:500) to stress-test the loss landscape.
2. **Deterministic Preprocessing:** Resizes raw field captures to $224 \times 224 \times 3$, scales intensities to $[0.0, 1.0]$, partitions into Stratified 5-Fold Cross-Validation splits, and isolates an independent balanced holdout test set (200 images: 100 Healthy / 100 Diseased).
3. **Dense Feature Reuse:** DenseNet121 provides deep feature reuse via direct skip connections across four dense blocks, reducing parameters to ~7.03M to combat overfitting, followed by GAP, Dropout ($p = 0.2$), and Softmax head.
4. **Branching Comparative Loss Regimes:**
   - **Branch A (Baseline):** Standard unweighted Categorical Cross-Entropy subject to majority class domination.
   - **Branch B (Stress Test):** Dynamically modulated Focal Loss ($\gamma = 2.0, \alpha = 0.25$) to test empirical survivability boundaries.
5. **Dual Evaluation & Forensic Outputs:**
   - **Quantitative Metrics:** Macro F1-score, Precision, Recall, fold variance, and Paired t-test/Wilcoxon tests to pinpoint the exact statistical breaking point.
   - **Forensic Grad-CAM Heatmaps:** Visual proof tracking whether spatial attention remains focused on foliar necrotic lesions or degrades into background leaf margins and specular reflections.

---

### Key Architectural Design Justifications (Literature-Grounded)

| Component / Choice | Selected Strategy | Rejected Alternative | Literature Grounding & Justification |
| :--- | :--- | :--- | :--- |
| **Feature Extraction Backbone** | **DenseNet121** | VGG16, ResNet50, YOLOv7 | Deng & Photong (2024) [1] and Raj et al. (2024) [4] proved DenseNet yields superior feature reuse in tea foliar classification with only ~7.03M parameters. VGG16 (138M) and ResNet50 (25.6M) cause severe overfitting under 1:500 starvation. |
| **Optimization Algorithm** | **Adam ($\eta = 10^{-4}$)** | Standard SGD, RMSprop | Kingma & Ba (2015) established that adaptive first and second moment estimation ($m_t, v_t$) maintains stable weight updates when minority gradient signals are sparse. |
| **Validation Strategy** | **Stratified 5-Fold CV + Test Holdout** | Single Train/Test Split | Satisfies Module IT41043 requirements by preventing single-split evaluation artifacts and measuring variance across cross-validation folds. |
| **Stress-Testing Loss Function** | **Focal Loss ($\gamma=2.0, \alpha=0.25$)** | Weighted Cross-Entropy | Lin et al. (2020) [5] proved $(1 - p_t)^\gamma$ dynamically down-weights easy majority examples, isolating the mathematical threshold where minority protection collapses. |
| **Explainable AI (XAI) Probe** | **Grad-CAM (`conv5_block16_concat`)** | LIME, Saliency Maps | Oad et al. (2024) [3] and Ozturk et al. (2025) [11] verified Grad-CAM on dense penultimate layers preserves high-resolution spatial localization of tea disease necrosis. |
