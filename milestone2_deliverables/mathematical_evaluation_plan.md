# Mathematical Evaluation Plan: Metric Invariance & Algorithmic Failure Analysis Under Extreme Data Starvation

**Project Title:** Visually Mapping the Algorithmic Breaking Point: Grad-CAM Explainability of Focal Loss Failure Under Extreme Data Starvation in Sri Lankan Tea Leaf Disease Detection  
**Module:** IT41043 - Intelligent Systems | Milestone 2 Submission  
**Authors:** E. Dasun Manjitha (`ITBIN-2313-0062`) & A. G. S. Lakshan (`ITBIN-2313-0055`)  
**Institution:** Faculty of IT, Horizon Campus  

---

## 1. The Fallacy of Raw Accuracy Under Extreme Imbalance

In standard classification tasks with balanced class distributions, Classification Accuracy is commonly adopted as an intuitive metric:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Where:
- $TP$: True Positives
- $TN$: True Negatives
- $FP$: False Positives
- $FN$: False Negatives

### The Trivial Classifier Hazard
Under the **Severe (1:100)** and **Extreme (1:500)** data starvation regimes explored in this research, standard accuracy becomes mathematically deceptive and clinically dangerous.

Consider an extreme distribution where a dataset contains $N_{\text{majority}} = 500$ healthy tea leaf samples and $N_{\text{minority}} = 1$ diseased leaf sample ($N_{\text{total}} = 501$). A degenerate or "blind" neural network that systematically predicts the majority class ("Healthy Leaf") for every single test input without extracting any meaningful visual features achieves:

$$\text{Accuracy}_{\text{trivial}} = \frac{500 + 0}{500 + 0 + 0 + 1} = \frac{500}{501} \approx 99.80\%$$

Despite a near-perfect accuracy score of **99.80%**, the model possesses **zero diagnostic capacity** ($FN = 1, TP = 0$), completely failing to detect crop pathogens in the field. In agricultural operations across Sri Lankan tea plantations, deploying a model evaluated purely on accuracy would induce **"plant blindness,"** allowing sporadic foliar outbreaks (such as *Algal Leaf Spot*, *Brown Blight*, or *Grey Blight*) to spread unabated, causing catastrophic economic crop yield degradation.

Therefore, accuracy cannot serve as a reliable objective function or benchmark in this study.

---

## 2. Mathematical Formulations of Diagnostic Metrics

To capture algorithmic degradation truthfully when minority classes are severely starved, we employ non-deceptive threshold-sensitive and class-specific metrics:

### 2.1 Precision (Positive Predictive Value)
Precision measures the fidelity of the model's positive disease predictions—specifically, the probability that an image flagged as diseased actually contains the foliar pathogen:

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **High Precision:** Minimizes False Positives ($FP$), ensuring agricultural estate managers do not incur unwarranted operational expenses by spraying fungicides on healthy bushes.
- **Limitation under Starvation:** A model predicting disease only once with high certainty can yield $\text{Precision} = 1.0$ while failing to detect dozens of actual diseased specimens ($FN \gg 0$).

---

### 2.2 Recall (Sensitivity or True Positive Rate)
Recall measures the model's completeness—specifically, the fraction of all true disease instances correctly identified:

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **High Recall:** Minimizes False Negatives ($FN$), ensuring diseased leaves do not slip through undetected.
- **Sensitivity to Collapse:** When an algorithm collapses under starvation, the minority-class $FN$ skyrockets while $TP \to 0$, driving Recall toward $0.00$.

---

### 2.3 The F1-Score (Harmonic Mean)
The F1-score provides the balanced harmonic mean between Precision and Recall. Unlike the arithmetic mean, the harmonic mean penalizes extreme imbalances between Precision and Recall severely:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

If a model collapses and yields a Recall of $0.0$, its $F_1$-score immediately drops to $0.0$, regardless of how many healthy leaves are correctly classified.

### 2.4 Macro-Averaged F1-Score ($F_1^{\text{macro}}$)
In multi-class disease diagnosis across $C$ distinct classes (where $c \in \{1, 2, \dots, C\}$):

$$F_1^{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F_{1, c}$$

Because each class is given equal mathematical weight regardless of its sample count, the Macro F1-score prevents the dominant healthy class from masking the total breakdown of the starved minority disease classes.

---

## 3. Mathematical Foundations of Loss Functions Under Stress

### 3.1 Standard Categorical Cross-Entropy (CCE)
For an input with ground-truth one-hot vector $y$ and predicted probability distribution $p$:

$$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} y_c \log(p_c)$$

Under severe class starvation ($N_{\text{majority}} \gg N_{\text{minority}}$), the cumulative gradient vector is completely swamped by easy negative/majority samples:

$$\sum_{i=1}^{N_{\text{total}}} \nabla_\theta \mathcal{L}_{\text{CE}}^{(i)} \approx \sum_{j \in \text{Majority}} \nabla_\theta \mathcal{L}_{\text{CE}}^{(j)}$$

The optimizer minimizes global empirical loss by simply shifting decision boundaries away from minority classes.

### 3.2 Focal Loss Formulation
Focal Loss introduces a dynamic modulating factor $(1 - p_t)^\gamma$ and weighting factor $\alpha_t$:

$$\mathcal{L}_{\text{FL}}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t$ is the model's estimated probability for the ground-truth class.
- $\gamma \ge 0$ is the tunable focusing parameter ($\gamma = 2.0$ in our benchmark).
- $\alpha_t \in [0, 1]$ addresses background/foreground imbalance ($\alpha = 0.25$).

When an easy sample is classified with high confidence ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, drastically suppressing its gradient contribution and focusing optimization on hard minority samples.

**The Central Hypothesized Breaking Point:**  
At extreme starvation ratios (e.g., 1:100 and 1:500), the sheer volume ratio of majority samples overcomes even the exponential modulating suppression of $(1 - p_t)^\gamma$, leading to gradient starvation and catastrophic visual attention collapse.

---

## 4. Empirical Validation Protocol: 5-Fold Stratified Cross-Validation

To ensure that reported $F_1$-scores and algorithmic failure points are statistically sound and not artifacts of a single fortunate or unfortunate random train-test split, we strictly enforce a **5-Fold Stratified Cross-Validation** protocol:

1. **Stratification Maintenance:** The original dataset $\mathcal{D}$ is partitioned into $K = 5$ disjoint folds $\{\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3, \mathcal{D}_4, \mathcal{D}_5\}$ such that the exact imbalance ratio (Baseline, 1:100, or 1:500) is preserved uniformly across every fold.
2. **Iterative Evaluation:** In each iteration $k \in \{1, \dots, 5\}$, fold $\mathcal{D}_k$ is reserved as the unseen test evaluation split while the remaining 4 folds ($\bigcup_{j \ne k} \mathcal{D}_j$) are used for model training and validation.
3. **Statistical Reporting:** Final reported F1-scores represent the mean and standard error across all 5 folds:
$$\mu_{F_1} = \frac{1}{5} \sum_{k=1}^{5} F_{1}^{(k)}, \quad \sigma_{F_1} = \sqrt{\frac{1}{5-1} \sum_{k=1}^{5} (F_{1}^{(k)} - \mu_{F_1})^2}$$

This rigorous statistical validation eliminates selection bias, providing irrefutable empirical proof of the exact mathematical ratio where Focal Loss collapses.
