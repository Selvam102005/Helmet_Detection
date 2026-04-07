"""
Detection Wrapper Module
=========================
A clean, CPU-friendly wrapper around the YOLOv8 model for inference.
"""

from typing import List, Dict
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    pass  # Allow importing for type hints without crashing if missing

import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.train_config import (
    YOLO_BEST_WEIGHTS,
    YOLO_CONF_THRESH,
    YOLO_IOU_THRESH,
    CLASS_MOTORCYCLE,
    CLASS_RIDER,
    CLASS_LICENSE_PLATE,
)


class ObjectDetector:
    def __init__(
        self,
        weights_path: str = str(YOLO_BEST_WEIGHTS),
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        device: str = "cpu"
    ):
        """
        Initialize the YOLOv8 detector.

        Args:
            weights_path: Path to trained .pt file
            conf_thresh: Minimum confidence score
            iou_thresh: NMS IoU threshold
            device: 'cpu' or 'cuda' (default 'cpu' for inference)
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path}. "
                "Train the model first or provide a valid path."
            )

        print(f"Loading YOLO detector from {weights_path} to {device}...")
        self.model = YOLO(weights_path)
        # Ensure model is on the right device
        self.model.to(device)


    def detect(self, frame) -> Dict[str, List[dict]]:
        # Run prediction
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )[0]

        detections = {"motorcycles": [], "riders": [], "plates": []}

        print(f"\n--- AI SENSE CHECK ---")
        print(f"Total objects found: {len(results.boxes)}")

        if not len(results.boxes):
            print("The AI sees ZERO objects. The image is likely too blurry or watermarked.")
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confs, classes):
            # LOG EVERY SINGLE BOX FOUND
            name = self.model.names[cls_id]
            print(f"   [FOUND] {name} (ID {cls_id}) with {conf:.4f} confidence")

            det = {"bbox": [float(x) for x in bbox], "conf": float(conf)}

            if cls_id == CLASS_MOTORCYCLE:
                detections["motorcycles"].append(det)
            elif cls_id == CLASS_RIDER:
                detections["riders"].append(det)
            elif cls_id == CLASS_LICENSE_PLATE:
                detections["plates"].append(det)

        return detections