# Research Architecture: Visually Mapping the Algorithmic Breaking Point of Focal Loss Under Extreme Data Starvation

```mermaid
flowchart TD
    %% Styling Definitions
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef prepStyle fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef modelStyle fill:#312e81,stroke:#a855f7,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef branch1Style fill:#450a0a,stroke:#f87171,stroke-width:2px,color:#fef2f2;
    classDef branch2Style fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5;
    classDef outputMetric fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#eff6ff;
    classDef outputGrad fill:#3b0764,stroke:#e879f9,stroke-width:2px,color:#fdf4ff;

    %% Data Ingestion Layer
    subgraph S1 ["1. Input Datasets (Controlled Imbalance Regimes)"]
        D1["Baseline Dataset<br/>(Natural Distribution: ~10,143 images<br/>7 Classes: Algal spot, Brown/Gray Blight, Helopeltis, etc.)"]:::inputStyle
        D2["Severe Starvation Dataset<br/>(1:100 Ratio: 1 Minority Diseased Leaf<br/>per 100 Majority Healthy Leaves)"]:::inputStyle
        D3["Extreme Starvation Dataset<br/>(1:500 Ratio: 1 Minority Diseased Leaf<br/>per 500 Majority Healthy Leaves)"]:::inputStyle
    end

    %% Preprocessing Pipeline
    subgraph S2 ["2. Deterministic Image Preprocessing Pipeline"]
        P1["Bilinear Spatial Resizing: 224 x 224 x 3"]:::prepStyle
        P2["Pixel Normalization: Pixel / 255.0 &rarr; [0.0, 1.0]"]:::prepStyle
        P3["K-Fold Partitioning: 5-Fold Stratified Cross-Validation"]:::prepStyle
        D1 --> P1
        D2 --> P1
        D3 --> P1
        P1 --> P2 --> P3
    end

    %% Feature Extractor Backbone
    subgraph S3 ["3. Deep Feature Extraction Backbone"]
        M1["Pre-trained DenseNet121 Architecture<br/>(Dense Blocks 1-4 with Transition Layers & Skip Connections)"]:::modelStyle
        M2["Global Average Pooling 2D (GAP) Layer<br/>Dimension Reduction: (7 &times; 7 &times; 1024) &rarr; (1024,)"]:::modelStyle
        M3["Dense Output Layer (Softmax Activation)<br/>Probability Distribution over C Classes"]:::modelStyle
        P3 --> M1 --> M2 --> M3
    end

    %% Dual Branching Loss Regimes
    subgraph S4 ["4. Branching Optimization & Stress Test Regimes"]
        B1["Branching Path 1: Standard Cross-Entropy Loss<br/>&bull; Equal weighting across all samples<br/>&bull; Vulnerable to majority class gradient swamping<br/>&bull; L_CE = -&Sigma; y_i log(p_i)"]:::branch1Style
        B2["Branching Path 2: Focal Loss Stress Test<br/>&bull; Dynamic modulating factor: (1 - p_t)^&gamma;<br/>&bull; Class balancing weight: &alpha; = 0.25<br/>&bull; Focusing parameter: &gamma; = 2.0<br/>&bull; L_FL = -&alpha;_t (1 - p_t)^&gamma; log(p_t)"]:::branch2Style
        M3 -->|"Ablation Line A"| B1
        M3 -->|"Ablation Line B"| B2
    end

    %% Dual Evaluation & Explainability Outputs
    subgraph S5 ["5. Diagnostic Evaluation & Explainable AI Outputs"]
        O1["Quantitative Metrics Suite<br/>&bull; Macro & Class-specific Precision, Recall<br/>&bull; Macro F1-Score & Minority F1-Score<br/>&bull; Algorithmic Breakdown Ratio Identification"]:::outputMetric
        O2["XAI Visual Heatmaps (tf-keras-vis Grad-CAM)<br/>&bull; Target Layer: Conv5_blk16_concat (DenseNet121)<br/>&bull; Spatial Attention Localization<br/>&bull; Visual Proof: Lesion Focus vs Background Hallucination"]:::outputGrad
        B1 --> O1
        B2 --> O1
        B1 --> O2
        B2 --> O2
    end
```

---

### Architectural Flow Summary
1. **Input Data Starvation Layer:** Accommodates the master Sri Lankan tea leaf dataset across three experimentally controlled regimes:
   - **Baseline:** Natural field distribution preserving original class frequencies.
   - **Severe (1:100):** Artificially starved to 1 diseased specimen per 100 healthy leaves.
   - **Extreme (1:500):** Ultra-sparse regime with 1 diseased specimen per 500 healthy leaves to stress-test the loss landscape.
2. **Standardized Preprocessing:** Resizes all raw RGB images to $224 \times 224$ pixels and scales color channels to $[0, 1]$, fed through 5-Fold Stratified Cross-Validation splits.
3. **Deep Feature Extraction Backbone:** DenseNet121 provides deep feature reuse via direct dense connections between layers, topped with GlobalAveragePooling2D and a Dense Softmax output layer.
4. **Branching Comparative Loss Paths:**
   - **Path 1 (Standard Categorical Cross-Entropy):** Baseline unweighted multi-class loss subject to majority class domination.
   - **Path 2 (Focal Loss Stress Test):** Modulated loss with focusing parameter $\gamma = 2.0$ and balancing factor $\alpha = 0.25$ designed to test empirical survivability boundaries.
5. **Dual Deliverable Output:**
   - **Macro F1-Score Metrics:** Reveals the exact numeric collapse threshold where minority class recognition fails.
   - **Grad-CAM Attention Maps:** Visual evidence demonstrating the shift of network attention from pathological foliar lesions to spurious background noise.
