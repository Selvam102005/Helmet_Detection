import cv2
import numpy as np
import datetime
from src.detector import ObjectDetector
from src.classifier import HelmetClassifier
from src.bbox_utils import crop_top_percent
from configs.train_config import HEAD_CROP_PERCENT

class SimpleHelmetPipeline:
    def __init__(self, yolo_weights, classifier_weights, device="cpu"):
        self.detector = ObjectDetector(weights_path=yolo_weights, device=device)
        self.classifier = HelmetClassifier(weights_path=classifier_weights, device=device)
        # OCR engine removed to focus only on helmets

    def process_image(self, frame):
        detections = self.detector.detect(frame)
        riders = detections.get("riders", [])
        
        report_data = [] 
        helmet_count = 0
        violation_count = 0

        for i, rider in enumerate(riders):
            rider_label = f"Rider {i+1}"
            rx1, ry1, rx2, ry2 = [max(0, int(v)) for v in rider["bbox"]]
            
            # 1. HELMET DETECTION
            head_crop = crop_top_percent(frame, rider["bbox"], HEAD_CROP_PERCENT)
            is_safe = False
            
            if head_crop is not None and head_crop.size > 0:
                label, _ = self.classifier.classify(head_crop)
                if label == "helmet":
                    is_safe = True

            if is_safe:
                helmet_count += 1
                color, status_msg = (0, 255, 0), "SAFE"
                report_data.append({
                    "Rider": rider_label,
                    "Status": "SAFE (Wearing Helmet)"
                })
            else:
                violation_count += 1
                color, status_msg = (0, 0, 255), "VIOLATION"
                report_data.append({
                    "Rider": rider_label,
                    "Status": "VIOLATION (No Helmet)"
                })

            # Draw UI on image
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 3)
            cv2.putText(frame, f"{rider_label}: {status_msg}", (rx1, ry1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, {"helmet": helmet_count, "no_helmet": violation_count, "details": report_data}