"""
Centralized Training Configuration
===================================
All hyperparameters, paths, and class mappings for the project.
OPTIMIZED FOR: 4GB Local NVIDIA GPU (GTX 1650/RTX 3050 etc.)
"""

import os
import torch
from pathlib import Path

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
# Matches the "Universal Merger" output folder
YOLO_DATA_DIR = DATA_DIR / "yolo_data" 
CLASSIFIER_DATA_DIR = DATA_DIR / "classifier_data"

# Config files
YOLO_DATASET_YAML = PROJECT_ROOT / "configs" / "yolo_dataset.yaml"

# Model weights output
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
YOLO_RUNS_DIR = PROJECT_ROOT / "models" / "yolo_runs"

# Output directory
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure critical directories exist
for d in [WEIGHTS_DIR, YOLO_RUNS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────
# Device Configuration
# ────────────────────────────────────────────────────────────
# Automatically uses your NVIDIA GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────
# YOLO Detection Configuration (Optimized for 4GB VRAM)
# ────────────────────────────────────────────────────────────
# Using Nano (n) model instead of Small (s) to fit in 4GB VRAM
YOLO_MODEL_VARIANT = "yolov8n.pt"   
YOLO_IMGSZ = 640

# LOWERED BATCH SIZE: Set to 4 to prevent "CUDA Out of Memory"
YOLO_BATCH_SIZE = 8                
YOLO_EPOCHS = 10                    # Good balance for 10k+ images
YOLO_PATIENCE = 5                  # Early stopping
YOLO_OPTIMIZER = "AdamW"
YOLO_LR0 = 0.001
YOLO_CONF_THRESH = 0.50       
YOLO_IOU_THRESH = 0.45              

# YOLO class names (Must match your Roboflow merger results)
YOLO_CLASS_NAMES = {
    0: "motorcycle",
    1: "rider",
    2: "license_plate",
}
YOLO_NUM_CLASSES = len(YOLO_CLASS_NAMES)

# Class IDs for lookup
CLASS_MOTORCYCLE = 0
CLASS_RIDER = 1
CLASS_LICENSE_PLATE = 2

# ────────────────────────────────────────────────────────────
# CNN Helmet Classifier Configuration (Optimized for 4GB VRAM)
# ────────────────────────────────────────────────────────────
# Using ResNet18 (lighter) instead of ResNet50 for 4GB GPU safety
CNN_MODEL_NAME = "resnet18"
CNN_IMGSZ = 224
CNN_BATCH_SIZE = 16                 # Lowered from 32 for 4GB RAM safety
CNN_EPOCHS = 30
CNN_LR = 1e-4
CNN_WEIGHT_DECAY = 1e-4
CNN_NUM_CLASSES = 2
CNN_CLASS_NAMES = {0: "helmet", 1: "no_helmet"}

# ImageNet normalization for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Best model weight filenames
CNN_BEST_WEIGHTS = WEIGHTS_DIR / "helmet_classifier_best.pt"
YOLO_BEST_WEIGHTS = WEIGHTS_DIR / "yolo_best.pt"

# ────────────────────────────────────────────────────────────
# Data Preparation
# ────────────────────────────────────────────────────────────
# Percentage of rider bbox to crop from the top for head isolation
HEAD_CROP_PERCENT = 0.25

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# ────────────────────────────────────────────────────────────
# Augmentation (Keep these as-is)
# ────────────────────────────────────────────────────────────
AUG_TARGET_PER_CLASS = 3000         
AUG_MOTION_BLUR_LIMIT = 7
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2

# ────────────────────────────────────────────────────────────
# Inference / Pipeline
# ────────────────────────────────────────────────────────────
HELMET_CONF_THRESHOLD = 0.60
PLATE_MATCH_IOU_THRESHOLD = 0.05