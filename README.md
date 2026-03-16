# Percieva Auto-Annotation Studio

Percieva Auto-Annotation Studio is a Streamlit workflow for building computer-vision datasets faster.

It combines frame extraction, model-gap filtering, augmentation, auto-annotation, evaluation, and metrics-grounded insights in one UI.

## What It Does

- Upload videos or image sets and extract frames into a working folder.
- Filter frames by comparing your latest custom model against a YOLO baseline.
- Split filtered results into:
  - `filtered_poor` for frames where the custom model is worse.
  - `filtered_other` for frames where it is the same or better.
- Copy matching `.txt` labels with filtered frames when they exist.
- Augment images before annotation.
- Run YOLO-based auto-annotation with class remapping.
- Compare models against ground truth with precision, recall, F1, and per-class analysis.
- Ask short, metrics-grounded questions in the Insights chat.

## App Pages

### Annotate
- Upload
- Filter
- Augment
- Image Gallery
- Auto-Annotate
- Annotated Gallery

### Model Comparison
- Evaluate saved models against `automatic_annotation/Model_Compare/ground_truth`
- Save outputs to `automatic_annotation/Model_Compare/output`
- Track run history in `automatic_annotation/Model_Compare/metrics.csv`

### Insights
- Chat over evaluation metrics
- Supports provider-backed responses using Groq Cloud or Cerebras Cloud
- Responses are intentionally short and to the point by default

## Recommended Workflow

1. Upload raw video or images.
2. Filter hard frames.
3. Review either split set in Image Gallery.
4. Augment selected data if needed.
5. Run Auto-Annotate.
6. Review annotations.
7. Run Model Comparison.
8. Use Insights to inspect weak classes and next actions.

## Quick Start

```bash
cd automatic_annotation
python -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in the terminal.

## Required Project Layout

```text
edgeverse_project/
├── automatic_annotation/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   ├── data_augmentation.py
│   ├── core/
│   ├── tools/
│   ├── class/
│   ├── videos/
│   ├── output_frames/
│   ├── output_annotation/
│   └── Model_Compare/
└── performance_testing/
```

## Important Folders

### `automatic_annotation/Model_Compare/`
- `new_model/` latest custom model candidates
- `yolo model/` YOLO baseline models
- `ground_truth/` evaluation images and YOLO labels
- `output/` evaluation visual outputs
- `metrics.csv` metrics history

### Runtime folders
- `output_frames/` extracted, uploaded, or augmented frames
- `output_annotation/` images, labels, and filtered output folders
- `videos/` uploaded source videos

## Key Scripts

- `automatic_annotation/streamlit_app.py` main application UI
- `automatic_annotation/tools/auto_annotation_runner.py` batch auto-annotation utility
- `automatic_annotation/tools/segment_video.py` large-video splitter
- `performance_testing/filter_frames_by_model_gap.py` frame filtering and evaluation utility
- `automatic_annotation/core/insights_chat.py` metrics-grounded chat backend

## API Keys For Insights

Set either of these before using Insights chat:

- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`

The app also loads values from `automatic_annotation/.env` when present.

## Notes

- Auto-annotation looks for one of these model files in `automatic_annotation/`:
  - `yolo12s.pt`
  - `yolo11n.pt`
  - `best.pt`
  - `yolo12m.pt`
- `class/new_classes.txt` is used to initialize `output_annotation/classes.txt` on first run.
- Filtering keeps paired label files when the source folder already contains matching `.txt` files.

## Need Full Setup Details?

See `guide.txt` for:

- exact folder expectations
- model placement
- first-run checklist
- troubleshooting
- modular file structure
