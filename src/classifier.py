"""
Classifier Wrapper Module
=========================
A clean, CPU-friendly wrapper around the ResNet-50 helmet classifier.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.train_config import (
    CNN_BEST_WEIGHTS,
    CNN_IMGSZ,
    CNN_NUM_CLASSES,
    CNN_CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class HelmetClassifier:
    def __init__(self, weights_path: str = str(CNN_BEST_WEIGHTS), device: str = "cpu"):
        """
        Initialize the ResNet-50 helmet classifier.

        Args:
            weights_path: Path to trained .pt file
            device: 'cpu' or 'cuda' (default 'cpu' for inference)
        """
        self.device = torch.device(device)

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Classifier weights not found at {weights_path}. "
                "Train the model first or provide a valid path."
            )

        print(f"Loading Helmet Classifier from {weights_path} to {device}...")

        # Build model architecture to match training
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, CNN_NUM_CLASSES),
        )

        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # Load class names if saved in checkpoint
            if "class_names" in checkpoint:
                self.class_names = {i: name for i, name in enumerate(checkpoint["class_names"])}
            else:
                self.class_names = CNN_CLASS_NAMES
        else:
            self.model.load_state_dict(checkpoint)
            self.class_names = CNN_CLASS_NAMES

        self.model = self.model.to(self.device)
        self.model.eval()

        # Input transforms matching training validation
        self.transform = transforms.Compose([
            transforms.Resize((CNN_IMGSZ, CNN_IMGSZ)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def classify(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classify a cropped head image.

        Args:
            crop: Numpy array (H, W, C) in BGR format (from OpenCV)

        Returns:
            Tuple: (predicted_class_name, confidence_score)
            Or (None, 0.0) if crop is invalid.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        # Convert BGR (OpenCV) to RGB (PIL/Torchvision)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # Apply transforms and add batch dimension
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)

            pred_idx = predicted.item()
            conf_val = conf.item()

            class_name = self.class_names.get(pred_idx, "unknown")

            return class_name, conf_val
