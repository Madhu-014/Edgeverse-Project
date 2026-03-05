"""Model-vs-ground-truth comparison runner for multiple `.pt` models.

This script evaluates one or more models against YOLO ground-truth labels,
writes visualized prediction frames, and appends metrics to `metrics.csv`.
"""

import numpy as np
import cv2
import time
import os
import csv
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import pandas as pd

# -------------------------------------------------
# LOAD CLASSES
# -------------------------------------------------
class_filename = "ground_truth/class/classes.txt"

class_list = []
with open(class_filename, 'r') as f:
    for line in f:
        class_list.append(line.strip())

label_dict = {i: name for i, name in enumerate(class_list)}

# -------------------------------------------------
# YOLO GT LOADER
# -------------------------------------------------
def load_yolo_gt(label_path, img_w, img_h):
    """Load YOLO txt labels and convert normalized boxes to pixel coordinates."""
    gt = []
    if not os.path.exists(label_path):
        return gt

    with open(label_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            xc *= img_w
            yc *= img_h
            bw *= img_w
            bh *= img_h

            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)

            gt.append([int(cls), x1, y1, x2, y2])
    return gt

# -------------------------------------------------
# IOU
# -------------------------------------------------
def iou(boxA, boxB):
    """Compute IoU for two `xyxy` boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def evaluate_folder(
    model,
    folder_path,
    output_dir,
    iou_thresh=0.5,
    conf_thresh=0.25
):
    """Evaluate a model on all JPG images in a folder and compute metrics.

    - Draws GT boxes in green and prediction boxes in red.
    - Tracks overall and per-class TP/FP/FN.
    - Returns metric dictionary for logging.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_classes = len(label_dict)

    # Overall stats
    TP = FP = FN = 0

    # Per-class stats
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in label_dict}

    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(folder_path, file)
        label_path = img_path.replace(".jpg", ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        gt_boxes = load_yolo_gt(label_path, w, h)

        # ---- Ultralytics prediction ----
        result = model(img, conf=conf_thresh, verbose=False)[0]

        preds = []
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            preds.append([cls, x1, y1, x2, y2, conf])

        used_gt = set()

        # ---- Draw Ground Truth (green) ----
        for cls, x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"GT:{label_dict[cls]}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # ---- Match predictions + draw Predicted boxes (red) ----
        for cls, x1, y1, x2, y2, conf in preds:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"P:{label_dict[cls]} {conf:.2f}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            best_iou = 0
            best_gt = -1
            for i, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if gt_cls != cls or i in used_gt:
                    continue
                current_iou = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = i

            if best_iou >= iou_thresh:
                TP += 1
                per_class[cls]["tp"] += 1
                used_gt.add(best_gt)
            else:
                FP += 1
                per_class[cls]["fp"] += 1

        # ---- Count FN for GT boxes not matched ----
        for i, (gt_cls, *_ ) in enumerate(gt_boxes):
            if i not in used_gt:
                FN += 1
                per_class[gt_cls]["fn"] += 1

        # ---- Save the image with GT and predicted boxes ----
        cv2.imwrite(os.path.join(output_dir, file), img)
        print(f"Processed {file}")

    # ---- Metrics ----
    overall_precision = TP / (TP + FP) if TP + FP else 0
    overall_recall = TP / (TP + FN) if TP + FN else 0

    metrics = {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall
    }

    for cid, stats in per_class.items():
        p = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0
        cls_name = label_dict[cid]
        metrics[f"precision_{cls_name}"] = p
        metrics[f"recall_{cls_name}"] = r

    return metrics


def log_metrics(csv_path, model_name, metrics):
    """Append one evaluation run to CSV, expanding columns when needed."""
    run_row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_row.update(metrics)

    # If CSV exists, load and expand schema if needed
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        for key in run_row.keys():
            if key not in df.columns:
                df[key] = ""

        df = pd.concat([df, pd.DataFrame([run_row])], ignore_index=True)
    else:
        df = pd.DataFrame([run_row])

    df.to_csv(csv_path, index=False)

# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":
    import sys
    
    # Get all available models from model folder
    model_dir = "model/"
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if not available_models:
        print("No models found in model/ folder!")
        sys.exit(1)
    
    # Allow selecting model via command line argument or process all
    if len(sys.argv) > 1:
        detection_model = sys.argv[1]
        if not detection_model.endswith('.pt'):
            detection_model += '.pt'
        models_to_process = [detection_model] if detection_model in available_models else []
    else:
        # Process all models
        models_to_process = available_models
    
    if not models_to_process:
        print(f"No valid models to process. Available: {available_models}")
        sys.exit(1)
    
    # Process each model
    for detection_model in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing model: {detection_model}")
        print(f"{'='*60}\n")
        
        model_path = model_dir + detection_model
        
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load model {detection_model}: {e}")
            continue
        
        model_name = str(detection_model).split(".")[0]
        output_predictions = "output/" + model_name
        
        print(f"Evaluating model on ground_truth/1 folder...")
        metrics = evaluate_folder(
            model,
            folder_path="ground_truth/1",
            output_dir=output_predictions
        )
        
        print(f"\nResults for {model_name}:")
        print(f"  Overall Precision: {metrics['overall_precision']:.4f}")
        print(f"  Overall Recall:    {metrics['overall_recall']:.4f}")
        
        # Print per-class metrics
        for key, val in metrics.items():
            if key.startswith('precision_') or key.startswith('recall_'):
                print(f"  {key}: {val:.4f}")
        
        log_metrics(
            csv_path="metrics.csv",
            model_name=model_name,
            metrics=metrics
        )
        
        print(f"\nMetrics saved to metrics.csv")
        print(f"Output images saved to {output_predictions}/")
    
    print(f"\n{'='*60}")
    print("All models processed successfully!")
    print(f"{'='*60}")


