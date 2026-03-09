"""Model-comparison metric utilities.

Contains replaceable metric implementations used by Model Comparison page.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def metric_safe_label(label: str) -> str:
    """Normalize class labels for metric-column keys."""
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def parse_yolo_annotation(txt_path: Path):
    """Parse YOLO annotation lines into tuples.

    Output tuple: (class_id, x_center, y_center, width, height)
    """
    boxes = []
    if not txt_path.exists():
        return boxes

    try:
        with open(txt_path, "r") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                parts = [float(value) for value in line.strip().split()]
                if len(parts) >= 5:
                    boxes.append((int(parts[0]), parts[1], parts[2], parts[3], parts[4]))
    except Exception:
        pass

    return boxes


def compare_annotations(gt_txt: Path, pred_txt: Path, img_shape, iou_threshold=0.5):
    """Compute class-aware TP/FP/FN using one-to-one IoU matching."""
    gt_boxes = parse_yolo_annotation(gt_txt)
    pred_boxes = parse_yolo_annotation(pred_txt)

    w, h = img_shape
    gt_pixel_boxes = []
    for class_id, x_c, y_c, bw, bh in gt_boxes:
        x1 = max(0, (x_c - bw / 2) * w)
        y1 = max(0, (y_c - bh / 2) * h)
        x2 = min(w, (x_c + bw / 2) * w)
        y2 = min(h, (y_c + bh / 2) * h)
        gt_pixel_boxes.append((class_id, x1, y1, x2, y2))

    pred_pixel_boxes = []
    for class_id, x_c, y_c, bw, bh in pred_boxes:
        x1 = max(0, (x_c - bw / 2) * w)
        y1 = max(0, (y_c - bh / 2) * h)
        x2 = min(w, (x_c + bw / 2) * w)
        y2 = min(h, (y_c + bh / 2) * h)
        pred_pixel_boxes.append((class_id, x1, y1, x2, y2))

    def calculate_iou(box1, box2):
        _, x1_1, y1_1, x2_1, y2_1 = box1
        _, x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        box2_area = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    candidates = []
    for gt_idx, gt_box in enumerate(gt_pixel_boxes):
        for pred_idx, pred_box in enumerate(pred_pixel_boxes):
            if gt_box[0] != pred_box[0]:
                continue
            iou = calculate_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    for _, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    per_class = {}
    for cls, *_ in gt_pixel_boxes:
        per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
    for cls, *_ in pred_pixel_boxes:
        per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})

    for gt_idx in matched_gt:
        cls_id = gt_pixel_boxes[gt_idx][0]
        per_class[cls_id]["tp"] += 1

    for pred_idx, pred_box in enumerate(pred_pixel_boxes):
        if pred_idx not in matched_pred:
            per_class[pred_box[0]]["fp"] += 1

    for gt_idx, gt_box in enumerate(gt_pixel_boxes):
        if gt_idx not in matched_gt:
            per_class[gt_box[0]]["fn"] += 1

    tp = len(matched_gt)
    fp = len(pred_pixel_boxes) - tp
    fn = len(gt_pixel_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "matches": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_class": per_class,
    }


def build_f1_series(df: pd.DataFrame, prec_col: str, rec_col: str):
    """Return F1 series using stored column when present, else derived values."""
    if "f1_score" in df.columns:
        f1_series = pd.to_numeric(df["f1_score"], errors="coerce")
    else:
        f1_series = pd.Series([np.nan] * len(df), index=df.index)

    precision = pd.to_numeric(df[prec_col], errors="coerce") if prec_col in df.columns else pd.Series([0.0] * len(df), index=df.index)
    recall = pd.to_numeric(df[rec_col], errors="coerce") if rec_col in df.columns else pd.Series([0.0] * len(df), index=df.index)
    derived = (2 * precision * recall) / (precision + recall)
    derived = derived.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return f1_series.fillna(derived)


def row_f1_value(row, prec_col: str, rec_col: str):
    """Return one-row F1, preferring stored f1_score if valid."""
    stored = row.get("f1_score", np.nan)
    if pd.notna(stored):
        return float(stored)

    precision = float(row.get(prec_col, 0.0)) if pd.notna(row.get(prec_col, np.nan)) else 0.0
    recall = float(row.get(rec_col, 0.0)) if pd.notna(row.get(rec_col, np.nan)) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def build_per_class_table(metrics_row):
    """Build sorted dataframe with class-level precision/recall/F1."""
    class_rows = []
    for col in metrics_row.index:
        if not str(col).startswith("precision_"):
            continue

        cls = str(col).replace("precision_", "")
        rec_col = f"recall_{cls}"
        if rec_col not in metrics_row.index:
            continue

        p_val = metrics_row.get(col, np.nan)
        r_val = metrics_row.get(rec_col, np.nan)
        if pd.isna(p_val) and pd.isna(r_val):
            continue

        p = float(p_val) if pd.notna(p_val) else 0.0
        r = float(r_val) if pd.notna(r_val) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        class_rows.append({
            "Class": cls.replace("_", " ").title(),
            "Precision": p,
            "Recall": r,
            "F1": f1,
        })

    if not class_rows:
        return pd.DataFrame()

    return pd.DataFrame(class_rows).sort_values("F1", ascending=False).reset_index(drop=True)
