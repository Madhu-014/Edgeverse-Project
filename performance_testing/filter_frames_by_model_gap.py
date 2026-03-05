"""Model evaluation and frame-filtering utilities.

This module provides two execution modes:
1) `evaluate`: evaluate one model against YOLO-format labels and report precision/recall.
2) `filter`: compare a custom model to a YOLO baseline and keep only hard frames where
    the custom model performs worse.

The Streamlit Filter tab calls this file in `filter` mode.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def _load_label_dict(class_filename="data/class/classes.txt"):
    """Load class index -> class name mapping from a classes.txt file."""
    class_path = Path(class_filename)
    if not class_path.exists():
        return {}

    class_list = []
    with open(class_path, "r") as file_obj:
        for line in file_obj:
            class_name = line.strip()
            if class_name:
                class_list.append(class_name)

    return {index: name for index, name in enumerate(class_list)}


label_dict = _load_label_dict()

def load_yolo_gt(label_path, img_w, img_h):
    """Read YOLO txt labels and convert normalized boxes to pixel coordinates."""
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


def iou(boxA, boxB):
    """Compute IoU between two `xyxy` boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def extract_boxes(result, conf_thresh=0.25):
    """Extract class, confidence, and xyxy boxes from one Ultralytics result."""
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    parsed = []
    for box, conf, cls in zip(boxes, confs, classes):
        if float(conf) < conf_thresh:
            continue
        parsed.append(
            {
                "box": box,
                "conf": float(conf),
                "cls": int(cls),
            }
        )

    return parsed


def custom_model_is_worse(
    custom_boxes,
    yolo_boxes,
    iou_thresh=0.4,
    max_allowed_box_diff=1,
):
    """Return True when custom detections are significantly worse than YOLO baseline.

    Logic summary:
    - if both detect nothing, frame is ignored;
    - if custom predicts more boxes than YOLO, it is treated as not worse;
    - if YOLO predicts sufficiently more boxes, custom is worse;
    - otherwise custom boxes must class-match YOLO boxes with IoU above threshold.
    """
    custom_count = len(custom_boxes)
    yolo_count = len(yolo_boxes)

    if custom_count == 0 and yolo_count == 0:
        return False

    if custom_count > yolo_count:
        return False

    if (yolo_count - custom_count) > max_allowed_box_diff:
        return True

    for custom_box in custom_boxes:
        best_iou = 0.0
        for yolo_box in yolo_boxes:
            if custom_box["cls"] != yolo_box["cls"]:
                continue
            best_iou = max(best_iou, iou(custom_box["box"], yolo_box["box"]))

        if best_iou < iou_thresh:
            return True

    return False


def _iter_image_files(folder_path):
    """List top-level image files in sorted order for deterministic processing."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    folder = Path(folder_path)
    if not folder.exists():
        return []

    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in image_exts
    ]
    return files


def filter_poor_frames(
    new_model_path,
    yolo_model_path,
    source_dir,
    destination_dir,
    conf_thresh=0.25,
    iou_thresh=0.4,
    max_allowed_box_diff=1,
    clear_destination=True,
):
    """Copy only poor-performing frames from source to destination.

    A frame is copied when `custom_model_is_worse(...)` returns True.
    Destination can be cleared first so it contains only selected frames.
    """
    source_path = Path(source_dir)
    destination_path = Path(destination_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    destination_path.mkdir(parents=True, exist_ok=True)

    if clear_destination:
        for file_path in _iter_image_files(destination_path):
            file_path.unlink(missing_ok=True)

    custom_model = YOLO(str(new_model_path))
    yolo_model = YOLO(str(yolo_model_path))

    image_files = _iter_image_files(source_path)
    selected_count = 0

    for image_path in image_files:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        custom_result = custom_model(frame, conf=conf_thresh, verbose=False)[0]
        yolo_result = yolo_model(frame, conf=conf_thresh, verbose=False)[0]

        custom_boxes = extract_boxes(custom_result, conf_thresh=conf_thresh)
        yolo_boxes = extract_boxes(yolo_result, conf_thresh=conf_thresh)

        is_worse = custom_model_is_worse(
            custom_boxes,
            yolo_boxes,
            iou_thresh=iou_thresh,
            max_allowed_box_diff=max_allowed_box_diff,
        )

        if is_worse:
            shutil.copy2(image_path, destination_path / image_path.name)
            selected_count += 1

    return {
        "source_dir": str(source_path),
        "destination_dir": str(destination_path),
        "total_images": len(image_files),
        "selected_images": selected_count,
        "ignored_images": max(0, len(image_files) - selected_count),
        "new_model_path": str(new_model_path),
        "yolo_model_path": str(yolo_model_path),
    }


def evaluate_folder(
    model,
    folder_path,
    output_dir,
    iou_thresh=0.5,
    conf_thresh=0.25
):
    """Evaluate one model against YOLO labels and save visualization frames.

    Returns dataset-level precision/recall via printed output.
    """
    os.makedirs(output_dir, exist_ok=True)

    TP = FP = FN = 0

    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(folder_path, file)
        label_path = os.path.join(
            folder_path, file.replace(".jpg", ".txt")
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        gt_boxes = load_yolo_gt(label_path, w, h)

        # ---- Ultralytics inference ----
        result = model(img, conf=conf_thresh, verbose=False)[0]

        preds = []
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            preds.append([cls, x1, y1, x2, y2, conf])

        used_gt = set()

        # ---- Draw GT (green) ----
        for cls, x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            gt_name = label_dict.get(cls, f"class_{cls}")
            cv2.putText(
                img, f"GT:{gt_name}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )

        # ---- Draw predictions + match ----
        for cls, x1, y1, x2, y2, conf in preds:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            pred_name = label_dict.get(cls, f"class_{cls}")
            cv2.putText(
                img, f"P:{pred_name} {conf:.2f}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1
            )

            best_iou = 0
            best_gt = -1

            for i, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if gt_cls != cls or i in used_gt:
                    continue

                current_iou = iou(
                    (x1, y1, x2, y2),
                    (gx1, gy1, gx2, gy2)
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = i

            if best_iou >= iou_thresh:
                TP += 1
                used_gt.add(best_gt)
            else:
                FP += 1

        FN += len(gt_boxes) - len(used_gt)

        cv2.imwrite(os.path.join(output_dir, file), img)
        print(f"Processed {file}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("\n===== DATASET METRICS =====")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")


def main():
    """CLI entrypoint for `evaluate` and `filter` workflows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "filter"], default="filter")
    parser.add_argument("--model", default="model/vapp_relu_320.pt")
    parser.add_argument("--folder-path", default="data/1")
    parser.add_argument("--output-dir", default="output/1")

    parser.add_argument("--new-model", default="")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--destination-dir", default="")

    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--max-allowed-box-diff", type=int, default=1)
    parser.add_argument("--clear-destination", action="store_true")

    args = parser.parse_args()

    if args.mode == "evaluate":
        model = YOLO(args.model)
        evaluate_folder(
            model,
            folder_path=args.folder_path,
            output_dir=args.output_dir,
            iou_thresh=args.iou_thresh,
            conf_thresh=args.conf_thresh,
        )
        return

    if not args.new_model or not args.yolo_model or not args.source_dir or not args.destination_dir:
        raise ValueError(
            "For filter mode, provide --new-model, --yolo-model, --source-dir, and --destination-dir"
        )

    summary = filter_poor_frames(
        new_model_path=args.new_model,
        yolo_model_path=args.yolo_model,
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        max_allowed_box_diff=args.max_allowed_box_diff,
        clear_destination=args.clear_destination,
    )

    print(json.dumps(summary))


if __name__ == "__main__":
    main()



						
					
					
					
					

	

