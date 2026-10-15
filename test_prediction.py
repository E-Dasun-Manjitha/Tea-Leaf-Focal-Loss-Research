"""
Quick Model Testing & Inference Script for Sri Lankan Tea Leaf Disease Detection
Authors: E. Dasun Manjitha & A. G. S. Lakshan

Allows you to test any tea leaf image against both the Baseline Model and the Extreme Starved (1:500) Model
to see the algorithmic breaking point in action!
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Classes
CLASS_NAMES = ["Diseased Leaf", "Healthy Leaf"]

def build_inference_model(weights_path):
    """Rebuilds DenseNet121 and loads pre-trained weights."""
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.2, name="dropout_reg")(x)
    outputs = Dense(2, activation="softmax", name="prediction_softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
    return model

def predict_single_image(image_path):
    """Feeds an image to both Baseline and Extreme 1:500 models and prints results side by side."""
    print("=" * 75)
    print(f"TESTING IMAGE: {image_path}")
    print("=" * 75)

    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' does not exist.")
        return

    # Load and preprocess
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    input_tensor = np.expand_dims(img_arr, axis=0)

    # 1. Baseline Model
    baseline_weights = os.path.join("models", "baseline_focal.weights.h5")
    if not os.path.exists(baseline_weights):
        baseline_weights = os.path.join("models", "baseline_cce.weights.h5")

    baseline_model = build_inference_model(baseline_weights)
    prob_base = baseline_model.predict(input_tensor, verbose=0)[0]
    pred_base = CLASS_NAMES[np.argmax(prob_base)]

    # 2. Extreme Starved Model (1:500)
    extreme_weights = os.path.join("models", "extreme_1_500_focal.weights.h5")
    if not os.path.exists(extreme_weights):
        extreme_weights = os.path.join("models", "extreme_1_500_cce.weights.h5")

    extreme_model = build_inference_model(extreme_weights)
    prob_ext = extreme_model.predict(input_tensor, verbose=0)[0]
    pred_ext = CLASS_NAMES[np.argmax(prob_ext)]

    print(f"\n[1] BASELINE MODEL (Normal Training):")
    print(f"    Prediction:      --> {pred_base.upper()} <--")
    print(f"    Confidence:      P(Diseased) = {prob_base[0]:.2%}, P(Healthy) = {prob_base[1]:.2%}")

    print(f"\n[2] EXTREME STARVED MODEL (1:500 Ratio - Starved of Diseased Data):")
    print(f"    Prediction:      --> {pred_ext.upper()} <--")
    print(f"    Confidence:      P(Diseased) = {prob_ext[0]:.2%}, P(Healthy) = {prob_ext[1]:.2%}")

    print("\n[DIAGNOSIS & OBSERVATION]:")
    if prob_base[0] > 0.5 and prob_ext[1] > 0.8:
        print("  => NOTICE THE FAILURE: The Baseline model accurately detects the diseased lesion,")
        print("     while the Extreme (1:500) model falsely claims it is 100% HEALTHY ('Plant Blindness')!")
        print("     This demonstrates the exact mathematical collapse of the algorithm.")
    print("=" * 75)

if __name__ == "__main__":
    # If user provided a specific image path via CLI:
    if len(sys.argv) > 1:
        img_file = sys.argv[1]
    else:
        # Default: Test on a sample diseased image from the test set
        test_diseased_dir = os.path.join("Test_Set", "Diseased")
        sample_images = [f for f in os.listdir(test_diseased_dir) if f.lower().endswith(('.jpg', '.png'))]
        if sample_images:
            img_file = os.path.join(test_diseased_dir, sample_images[0])
        else:
            img_file = ""

    if img_file:
        predict_single_image(img_file)
    else:
        print("No image provided and Test_Set/Diseased is empty.")
