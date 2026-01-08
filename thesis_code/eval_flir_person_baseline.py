import os
from PIL import Image

# === (1) INPUT PATHS ===

# - IMG_DIR: directory containing all validation images
#            (only images with at least one GT person)
# - GT_DIR:  directory with ground-truth labels
#            (YOLO format: class xc yc w h) -> 5 columns
# - PRED_DIR: directory with YOLO prediction labels
#             (YOLO format: class xc yc w h conf) -> 6 columns
IMG_DIR  = r"C:\stage defensie\FLIR_ADAS_1_3\val\thermal_8_bit_cat1_person"
GT_DIR   = r"C:\stage defensie\FLIR_ADAS_1_3\val\flir_subset_person_yolo_annotated_groundtruth\labels"
PRED_DIR = r"C:\stage defensie\flir_cat1_person_coco_baseline\pred_conf025\labels"

# === (2) EVALUATION SETTINGS ===
# IOU_THR = 0.5: standard object detection matching criterion (COCO / FLIR baseline)
# CLASS_ID = 0: 'person' class in COCO / YOLO (we evaluate persons only)
IOU_THR = 0.5
CLASS_ID = 0

def yolo_xywh_to_xyxy(xc, yc, w, h, W, H):
    """
    Convert YOLO normalized bounding boxes (xc, yc, w, h)
    to pixel-based corner format (x1, y1, x2, y2).
    """
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return (x1, y1, x2, y2)

def iou(a, b):
    """
    Compute the Intersection over Union (IoU) between two
    bounding boxes a and b in pixel coordinates.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def read_gt(txt_path, W, H):
    """
    Read ground-truth labels for a single image:
    - Expected format per line: 'class xc yc w h'
    - Filter by CLASS_ID (person)
    - Convert to pixel-based bounding boxes
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            if cls != CLASS_ID:
                continue
            xc, yc, w, h = map(float, p[1:5])
            boxes.append(yolo_xywh_to_xyxy(xc, yc, w, h, W, H))
    return boxes

def read_pred(txt_path, W, H):
    """
    Read YOLO prediction labels for a single image
    (generated using detect.py with --save-conf):
    - Expected format per line: 'class xc yc w h conf'
    - Filter by CLASS_ID (person)
    - Store bounding box and confidence score
    - Sort predictions by descending confidence so that
      higher-confidence detections are matched first
      (standard object detection evaluation practice)
    """
    preds = []
    if not os.path.exists(txt_path):
        return preds
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            cls = int(float(p[0]))
            if cls != CLASS_ID:
                continue
            xc, yc, w, h, conf = map(float, p[1:6])
            preds.append((yolo_xywh_to_xyxy(xc, yc, w, h, W, H), conf))
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds

# === (3) STATISTIC COUNTERS ===
# TP: true positives (correct matches)
# FP: false positives (detections without GT match)
# FN: false negatives (missed GT persons)
TP = FP = FN = 0
n_gt = n_pred = 0

# Collect all validation images
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

missing_gt = 0
missing_pred = 0

# === (4) LOOP OVER ALL IMAGES ===
for img_name in images:

    # Extract filename stem (e.g. FLIR_09939.jpeg -> FLIR_09939)
    # Used to match images with corresponding label files
    stem = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)

    # Open image to obtain width and height (required for coordinate conversion)
    with Image.open(img_path) as im:
        W, H = im.size

    # Construct paths to GT and prediction label files
    gt_path = os.path.join(GT_DIR, stem + ".txt")
    pr_path = os.path.join(PRED_DIR, stem + ".txt")

    # If GT is missing, skip image (cannot be evaluated)
    if not os.path.exists(gt_path):
        missing_gt += 1
        continue

    # If prediction is missing, all GT persons are counted as false negatives
    if not os.path.exists(pr_path):
        missing_pred += 1
        gt_boxes = read_gt(gt_path, W, H)
        n_gt += len(gt_boxes)
        FN += len(gt_boxes)
        continue

    # Read GT and prediction boxes
    gt_boxes = read_gt(gt_path, W, H)
    pr_boxes = read_pred(pr_path, W, H)

    n_gt += len(gt_boxes)
    n_pred += len(pr_boxes)

    # === (5) MATCHING: PREDICTIONS -> GROUND TRUTH ===

    # Keep track of already matched GT boxes
    # (each GT object can be matched to only one prediction)
    matched = set()

    # Iterate over predictions (sorted by confidence)
    for pb, _ in pr_boxes:
        best_iou, best_gi = 0.0, None

        # Find the GT box with highest IoU that is not yet matched
        for gi, gb in enumerate(gt_boxes):
            if gi in matched:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou, best_gi = v, gi

        # If IoU exceeds threshold, count as true positive
        # Otherwise, count as false positive
        if best_iou >= IOU_THR and best_gi is not None:
            TP += 1
            matched.add(best_gi)
        else:
            FP += 1

    # Remaining unmatched GT boxes are false negatives
    FN += (len(gt_boxes) - len(matched))

# === (6) METRIC COMPUTATION ===
# Precision = TP / (TP + FP): reliability of detections
# Recall    = TP / (TP + FN): proportion of GT persons detected
# F1-score  =  mean of precision and recall
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

# === (7) OUTPUT / REPORTING ===
print("=== BASELINE EVALUATION: YOLOv7 COCO -> FLIR val (person-only) ===")
print("IMG_DIR :", IMG_DIR)
print("GT_DIR  :", GT_DIR)
print("PRED_DIR:", PRED_DIR)
print(f"Images total: {len(images)} | missing GT: {missing_gt} | missing pred: {missing_pred}")
print(f"GT persons: {n_gt} | Pred persons: {n_pred}")
print(f"IoU >= {IOU_THR}")
print(f"TP={TP}  FP={FP}  FN={FN}")
print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
