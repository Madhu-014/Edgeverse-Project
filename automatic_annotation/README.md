# Percieva™ Auto-Annotation Studio

Percieva™ Auto-Annotation Studio is a Streamlit-based workflow for:
- ingesting videos/images,
- filtering hard frames using model-vs-YOLO comparison,
- augmenting data,
- auto-annotating with YOLO,
- comparing model performance,
- and tracking high-level insights.

---

## Core Features

### 1) Annotate Page
The Annotate page contains six tabs:

1. **Upload**
   - Upload a video or images (ZIP/files).
   - Extract frames at a selected interval.
   - Save frames into a target folder.

2. **Filter**
   - Uses the latest model from `Model_Compare/new_model`.
   - Uses YOLO baseline model from `Model_Compare/yolo model`.
   - Compares both models frame-by-frame.
   - Keeps only frames where the latest model performs worse.
   - Writes selected frames to destination folder (default: `output_annotation`).

3. **Augment**
   - Applies augmentation variants to source images.
   - Supports noise, blur, motion blur, brightness/contrast, rotation, fog.

4. **Image Gallery**
   - Browse uploaded/extracted images.
   - Pagination and delete support.

5. **Auto-Annotate**
   - Runs YOLO auto-annotation over selected frames.
   - Writes labels in YOLO TXT format.
   - Includes class management through `classes.txt`.

6. **Annotated Gallery**
   - Visual review of annotations with rendered bounding boxes.
   - Delete image + matching label together.

---

### 2) Model Comparison Page
This page now combines **comparison + analytics** in one place.

- Run evaluation for available models against `Model_Compare/ground_truth`.
- Save visualized comparison outputs.
- Append metrics to `Model_Compare/metrics.csv`.
- View:
  - overall precision/recall/F1,
  - per-class metrics,
  - comparison frames,
  - metrics trend/history,
  - multi-model comparison tables/charts.

---

### 3) Insights Page
A lightweight summary dashboard:
- frame count,
- annotated image count,
- label file count,
- model run count,
- latest run summary (model, precision, recall).

---

## Project Pipeline (Recommended)

1. **Upload** raw video/images in Annotate → Upload.
2. **Filter** hard frames in Annotate → Filter.
3. **Augment** selected data in Annotate → Augment (optional).
4. **Auto-Annotate** in Annotate → Auto-Annotate.
5. **Review** in Annotate → Annotated Gallery.
6. **Evaluate** model quality in Model Comparison.
7. **Track status** in Insights.

---

## Folder Expectations

Inside `automatic_annotation/Model_Compare`:

- `new_model/` → your latest candidate model(s), `.pt`
- `yolo model/` → YOLO baseline model(s), `.pt`
- `ground_truth/` → evaluation images + YOLO labels
- `output/` → generated comparison visualization outputs
- `metrics.csv` → cumulative run metrics

Main runtime folders:

- `output_frames/` → extracted/uploaded frames
- `output_annotation/` → annotation labels/images and default filter destination
- `videos/` → uploaded video files

---

## Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) GPU-enabled PyTorch for faster inference

### Setup

```bash
cd automatic_annotation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Run Instructions

### Start Streamlit UI

```bash
cd automatic_annotation
source ../venv/bin/activate  # or your active environment
streamlit run streamlit_app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

### Optional: Run performance filter utility directly

```bash
cd ..
python performance_testing/filter_frames_by_model_gap.py \
  --mode filter \
  --new-model automatic_annotation/Model_Compare/new_model/<latest_model>.pt \
  --yolo-model "automatic_annotation/Model_Compare/yolo model/<yolo_model>.pt" \
  --source-dir automatic_annotation/output_frames \
  --destination-dir automatic_annotation/output_annotation \
  --conf-thresh 0.25 \
  --iou-thresh 0.40 \
  --clear-destination
```

### Optional: Run evaluation mode directly

```bash
cd ..
python performance_testing/filter_frames_by_model_gap.py \
  --mode evaluate \
  --model performance_testing/model/vapp_relu_320.pt \
  --folder-path performance_testing/data/1 \
  --output-dir performance_testing/output/1 \
  --conf-thresh 0.25 \
  --iou-thresh 0.50
```

---

## Threshold Tuning (Filter Tab)

- **Confidence threshold**
  - Higher → fewer detections considered (stricter).
  - Lower → more detections considered (looser).

- **IoU threshold**
  - Higher → tighter overlap required to count as a match.
  - Lower → looser overlap accepted.

Practical effect: increasing either threshold typically marks more frames as poor.

---

## Key Files

- `streamlit_app.py` → main UI and workflows
- `auto_annotation.py` → batch auto-label generation
- `data_augmentation.py` → augmentation transforms
- `Model_Compare/evaluate_models_against_ground_truth.py` → model-ground-truth evaluation + metrics logging
- `../performance_testing/filter_frames_by_model_gap.py` → evaluate/filter CLI used by Filter tab

---

## Troubleshooting

- **No models found in Filter tab**
  - Ensure `.pt` files exist in:
    - `Model_Compare/new_model`
    - `Model_Compare/yolo model`

- **No frames selected by filter**
  - Lower IoU and/or confidence thresholds.
  - Verify source folder contains images.

- **Metrics not showing in Model Comparison**
  - Run at least one evaluation from the page.
  - Confirm `Model_Compare/metrics.csv` is writable.

- **Import errors in editor**
  - Activate your virtual environment and install requirements.

---

## Notes

- Filter currently copies selected image files to destination.
- If you want recursive copy with source subfolder preservation or paired `.txt` label copy, that can be added.
