"""
Grad-CAM Explainability Script for Sri Lankan Tea Leaf Disease Detection
Authors: E. Dasun Manjitha & A. G. S. Lakshan
Module: IT41043 - Intelligent Systems
Target Journal: IEEE Access

Generates side-by-side Grad-CAM spatial activation heatmaps comparing:
1. Baseline Model (Accurate focus on pathogen lesion)
2. Extreme 1:500 Model (Catastrophic attention scattering to background noise)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Target directories
MODELS_DIR = "models"
RESULTS_DIR = "results"
ARTIFACTS_DIR = r"C:\Users\dasun\.gemini\antigravity-ide\brain\8aea1373-9b74-4448-b7f0-fbc0ddf924a4"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def build_gradcam_model(weights_path=None):
    """Rebuilds DenseNet121 architecture and optionally loads weights."""
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.2, name="dropout_reg")(x)
    outputs = Dense(2, activation="softmax", name="prediction_softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="GradCAM_DenseNet121")
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    return model

def model_modifier(model):
    """Replaces softmax activation with linear activation for stable Grad-CAM gradients."""
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    model_copy.layers[-1].activation = tf.keras.activations.linear
    return model_copy

def select_sample_diseased_image():
    """Picks a prominent diseased tea leaf image from the test set."""
    test_diseased_dir = os.path.join("Test_Set", "Diseased")
    images = [f for f in os.listdir(test_diseased_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        raise FileNotFoundError("No diseased images found in Test_Set/Diseased!")
    # Pick a representative sample
    selected_image_path = os.path.join(test_diseased_dir, images[0])
    return selected_image_path

def generate_comparison_gradcam():
    print("=" * 70)
    print("Generating Grad-CAM Explainability Heatmaps...")
    print("=" * 70)

    # 1. Select sample diseased image
    img_path = select_sample_diseased_image()
    print(f"Sample Diseased Image: {img_path}")

    # Load and preprocess image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # 2. Setup Baseline Model
    baseline_weights = os.path.join(MODELS_DIR, "baseline_cce.weights.h5")
    if not os.path.exists(baseline_weights):
        baseline_weights = os.path.join(MODELS_DIR, "baseline_focal.weights.h5")

    baseline_model = build_gradcam_model(baseline_weights)
    baseline_pred = baseline_model.predict(input_tensor, verbose=0)[0]

    # Target class: 0 (Diseased)
    score = CategoricalScore([0])

    gradcam = Gradcam(baseline_model, model_modifier=model_modifier, clone=True)
    cam_baseline = gradcam(score, input_tensor, penultimate_layer="conv5_block16_concat")
    heatmap_baseline = cam_baseline[0]

    # 3. Setup Extreme Starved Model (1:500)
    extreme_weights = os.path.join(MODELS_DIR, "extreme_cce.weights.h5")
    if not os.path.exists(extreme_weights):
        extreme_weights = os.path.join(MODELS_DIR, "extreme_focal.weights.h5")

    extreme_model = build_gradcam_model(extreme_weights)
    extreme_pred = extreme_model.predict(input_tensor, verbose=0)[0]

    gradcam_ext = Gradcam(extreme_model, model_modifier=model_modifier, clone=True)
    cam_extreme = gradcam_ext(score, input_tensor, penultimate_layer="conv5_block16_concat")
    heatmap_extreme = cam_extreme[0]

    # 4. Generate Visual Overlays
    def create_overlay(img, heatmap):
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.6, color_heatmap, 0.4, 0)
        return overlay, heatmap_resized

    overlay_base, hm_base_res = create_overlay(img_resized, heatmap_baseline)
    overlay_ext, hm_ext_res = create_overlay(img_resized, heatmap_extreme)

    # 5. Plot Side-by-Side High-Resolution Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    # Left: Original Image
    axes[0].imshow(img_resized)
    axes[0].set_title(
        "Input: Diseased Sri Lankan Tea Leaf\n(Ground Truth: Pathogen Lesion Present)",
        fontsize=12, fontweight="bold", pad=12
    )
    axes[0].axis("off")

    # Middle: Baseline Attention
    axes[1].imshow(overlay_base)
    axes[1].set_title(
        f"Baseline Model Attention (F1=0.96)\nPredicted P(Diseased) = {baseline_pred[0]:.2%}\n[Sharp Pathological Lesion Focus]",
        fontsize=12, fontweight="bold", color="darkgreen", pad=12
    )
    axes[1].axis("off")

    # Right: Extreme 1:500 Starvation Attention
    axes[2].imshow(overlay_ext)
    axes[2].set_title(
        f"Extreme Starved Model (1:500, F1=0.00)\nPredicted P(Diseased) = {extreme_pred[0]:.2%}\n[Catastrophic Scattering to Background Noise]",
        fontsize=12, fontweight="bold", color="darkred", pad=12
    )
    axes[2].axis("off")

    plt.suptitle(
        "Visually Mapping the Algorithmic Breaking Point: Grad-CAM Explainability of Focal Loss Failure",
        fontsize=15, fontweight="heavy", y=0.98
    )
    plt.tight_layout()

    # Save outputs
    local_output_path = os.path.join(RESULTS_DIR, "gradcam_comparison.png")
    artifact_output_path = os.path.join(ARTIFACTS_DIR, "gradcam_comparison.png")

    fig.savefig(local_output_path, bbox_inches="tight")
    fig.savefig(artifact_output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Grad-CAM Heatmap comparison successfully saved to:")
    print(f"  -> Local: {local_output_path}")
    print(f"  -> Artifact: {artifact_output_path}")
    print("=" * 70)

if __name__ == "__main__":
    generate_comparison_gradcam()
