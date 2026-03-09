"""Gallery and image preview helpers.

UI can swap gallery behavior by replacing these functions without changing
Streamlit tab structure.
"""

from pathlib import Path
import os
import re

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def natural_sort_key(path: Path):
    """Sort names with numeric awareness (frame2 before frame10)."""
    name = path.name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def list_images_recursive(dir_path: Path) -> list[Path]:
    """Collect all image files recursively in natural sort order."""
    images = []
    if dir_path.exists():
        for root, _, files in os.walk(dir_path):
            for file_name in sorted(files):
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    images.append(Path(root) / file_name)
    return sorted(images, key=natural_sort_key)


def load_image_pil_rgb(path: Path):
    """Load image path as RGB PIL image; return None if unreadable."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def draw_yolo_boxes_from_txt(img_pil: Image.Image, txt_path: Path) -> Image.Image:
    """Render YOLO boxes from txt annotation onto a PIL image."""
    img = np.array(img_pil).copy()
    h, w = img.shape[:2]

    if txt_path.exists():
        try:
            with open(txt_path, "r") as file_obj:
                lines = [line.strip() for line in file_obj if line.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            pass

    return Image.fromarray(img)
