"""
Data Engineering & Synthetic Starvation Script for Sri Lankan Tea Leaf Disease Detection
Author: E. Dasun Manjitha & A. G. S. Lakshan
Module: IT41043 - Intelligent Systems
Project: Visually Mapping the Algorithmic Breaking Point of Focal Loss Under Extreme Data Starvation
"""

import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set seed for reproducible starvation sampling
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MASTER_DATASET_DIR = "Merged_Tea_Leaf_Dataset"
OUTPUT_DIRS = {
    "baseline": "Baseline",
    "severe": "Severe_1_100",
    "extreme": "Extreme_1_500",
    "test": "Test_Set"
}

DISEASE_CLASSES = [
    "1. Tea algal leaf spot",
    "2. Brown Blight",
    "3. Gray Blight",
    "4. Helopeltis",
    "5. Red spider",
    "6. Green mirid bug"
]
HEALTHY_CLASS = "7. Healthy leaf"

def collect_image_paths(base_dir):
    """Gathers all image paths for healthy and diseased leaves."""
    healthy_paths = []
    diseased_paths = []

    healthy_dir = os.path.join(base_dir, HEALTHY_CLASS)
    if os.path.exists(healthy_dir):
        for f in os.listdir(healthy_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                healthy_paths.append(os.path.join(healthy_dir, f))

    for d_cls in DISEASE_CLASSES:
        d_dir = os.path.join(base_dir, d_cls)
        if os.path.exists(d_dir):
            for f in os.listdir(d_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    diseased_paths.append(os.path.join(d_dir, f))

    random.shuffle(healthy_paths)
    random.shuffle(diseased_paths)
    return healthy_paths, diseased_paths

def copy_images(file_list, target_dir):
    """Copies list of image files to destination folder."""
    os.makedirs(target_dir, exist_ok=True)
    for src in file_list:
        dst = os.path.join(target_dir, os.path.basename(src))
        shutil.copy2(src, dst)

def create_starvation_regimes():
    """Generates Baseline, Severe (1:100), and Extreme (1:500) datasets."""
    print("=" * 70)
    print("Executing Data Starvation Engineering Pipeline...")
    print("=" * 70)

    healthy_all, diseased_all = collect_image_paths(MASTER_DATASET_DIR)
    print(f"Total Available Healthy Images: {len(healthy_all)}")
    print(f"Total Available Diseased Images: {len(diseased_all)}")

    # 1. Carve out a balanced unseen Test Set for consistent, unbiased evaluation
    test_healthy = healthy_all[:100]
    test_diseased = diseased_all[:100]

    # Remaining pool for training/starvation experiments
    healthy_pool = healthy_all[100:]
    diseased_pool = diseased_all[100:]

    # Clean existing generated directories if present
    for d in OUTPUT_DIRS.values():
        if os.path.exists(d):
            shutil.rmtree(d)

    # 2. Build Test Set
    copy_images(test_healthy, os.path.join(OUTPUT_DIRS["test"], "Healthy"))
    copy_images(test_diseased, os.path.join(OUTPUT_DIRS["test"], "Diseased"))
    print(f"\n[Test Set Created] Healthy: {len(test_healthy)}, Diseased: {len(test_diseased)}")

    # 3. Build Baseline Dataset (Balanced control distribution: 1000 Healthy, 1000 Diseased)
    baseline_healthy = healthy_pool[:1000]
    baseline_diseased = diseased_pool[:1000]
    copy_images(baseline_healthy, os.path.join(OUTPUT_DIRS["baseline"], "Healthy"))
    copy_images(baseline_diseased, os.path.join(OUTPUT_DIRS["baseline"], "Diseased"))
    print(f"[Baseline Created] Healthy: {len(baseline_healthy)}, Diseased: {len(baseline_diseased)} (Natural Balance 1:1)")

    # 4. Build Severe Starvation Dataset (1:100 Ratio -> Exactly 1 Diseased leaf for every 100 Healthy leaves)
    severe_healthy = healthy_pool[:1000]
    severe_diseased = diseased_pool[:10]  # 10 / 1000 = 1:100
    copy_images(severe_healthy, os.path.join(OUTPUT_DIRS["severe"], "Healthy"))
    copy_images(severe_diseased, os.path.join(OUTPUT_DIRS["severe"], "Diseased"))
    print(f"[Severe 1:100 Created] Healthy: {len(severe_healthy)}, Diseased: {len(severe_diseased)} (Exact Ratio 1:100)")

    # 5. Build Extreme Starvation Dataset (1:500 Ratio -> Exactly 1 Diseased leaf for every 500 Healthy leaves)
    extreme_healthy = healthy_pool[:1000]
    extreme_diseased = diseased_pool[:2]   # 2 / 1000 = 1:500
    copy_images(extreme_healthy, os.path.join(OUTPUT_DIRS["extreme"], "Healthy"))
    copy_images(extreme_diseased, os.path.join(OUTPUT_DIRS["extreme"], "Diseased"))
    print(f"[Extreme 1:500 Created] Healthy: {len(extreme_healthy)}, Diseased: {len(extreme_diseased)} (Exact Ratio 1:500)")

    print("\nAll datasets partitioned and written successfully.")
    print("=" * 70)

def get_data_generators(target_dir, batch_size=32, is_training=True):
    """
    Builds standard Keras ImageDataGenerator resizing to 224x224 and normalizing pixels by 1/255.
    """
    if is_training:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    generator = datagen.flow_from_directory(
        target_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=is_training,
        seed=SEED
    )
    return generator

if __name__ == "__main__":
    create_starvation_regimes()

    print("\nVerifying ImageDataGenerator preprocessing (224x224, 1/255 normalization)...")
    for name, path in OUTPUT_DIRS.items():
        gen = get_data_generators(path, batch_size=16, is_training=False)
        batch_x, batch_y = next(gen)
        print(f"  [{name.upper()}] Generator Output Shape: Images={batch_x.shape}, Labels={batch_y.shape}, Min={batch_x.min():.2f}, Max={batch_x.max():.2f}")
    print("\nImageDataGenerator verification complete!")
