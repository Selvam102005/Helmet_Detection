"""
Bounding Box Utilities
=======================
Mathematical operations for bounding boxes: computing IoU, calculating
containment, cropping regions, and matching riders to license plates.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Boxes must be in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    if inter_area == 0:
        return 0.0

    # Compute area of both boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def is_center_inside(outer_box: List[float], inner_box: List[float]) -> bool:
    """
    Check if the center of the inner_box is inside the outer_box.
    Used as a fallback heuristic if IoU is very low.
    """
    inner_cx = (inner_box[0] + inner_box[2]) / 2.0
    inner_cy = (inner_box[1] + inner_box[3]) / 2.0

    return (
        outer_box[0] <= inner_cx <= outer_box[2] and
        outer_box[1] <= inner_cy <= outer_box[3]
    )


def crop_top_percent(
    image: np.ndarray,
    bbox: List[int],
    percent: float = 0.25
) -> Optional[np.ndarray]:
    """
    Crop the top N% of a bounding box from the image (used for head extraction).

    Args:
        image: Original frame (H, W, C)
        bbox: [x1, y1, x2, y2] pixel coordinates
        percent: Fraction to extract from the top

    Returns:
        Cropped numpy array, or None if invalid.
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = image.shape[:2]

    # Clamp to image boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))

    h = y2 - y1
    w = x2 - x1

    if h < 10 or w < 10:
        return None  # Box too small

    # Calculate top crop
    crop_h = int(h * percent)
    y_crop_end = y1 + crop_h

    return image[y1:y_crop_end, x1:x2]


def match_riders_to_motorcycles(
    riders: List[dict],
    motorcycles: List[dict],
    iou_thresh: float = 0.01
) -> Dict[int, int]:
    """
    Match riders to motorcycles based on IoU.

    Returns:
        Dict mapping rider_index -> motorcycle_index
    """
    matches = {}
    matched_motos = set()

    for r_idx, rider in enumerate(riders):
        best_iou = 0
        best_m_idx = -1

        for m_idx, moto in enumerate(motorcycles):
            if m_idx in matched_motos:
                continue

            iou = compute_iou(rider["bbox"], moto["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_m_idx = m_idx

        if best_iou >= iou_thresh and best_m_idx != -1:
            matches[r_idx] = best_m_idx
            matched_motos.add(best_m_idx)

    return matches


def match_rider_to_plate(
    rider_idx: int,
    motorcycle_idx: Optional[int],
    riders: List[dict],
    motorcycles: List[dict],
    plates: List[dict],
) -> Optional[dict]:
    """
    Find the license plate corresponding to a rider.

    Strategy:
    1. If rider is matched to a motorcycle, find plate inside/overlapping the motorcycle.
    2. Fallback: find plate overlapping the rider's bbox.
    """
    if not plates:
        return None

    # Determine the "parent" box to look inside
    parent_box = None
    if motorcycle_idx is not None and motorcycle_idx < len(motorcycles):
        parent_box = motorcycles[motorcycle_idx]["bbox"]
    else:
        parent_box = riders[rider_idx]["bbox"]

    best_iou = 0
    best_plate = None

    # First pass: try standard IoU overlap
    for plate in plates:
        iou = compute_iou(parent_box, plate["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_plate = plate

    if best_plate and best_iou > 0.01:
        return best_plate

    # Second pass: check if plate center is inside parent box
    for plate in plates:
        if is_center_inside(parent_box, plate["bbox"]):
            return plate

    return None
