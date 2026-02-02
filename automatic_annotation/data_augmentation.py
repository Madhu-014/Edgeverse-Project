import os
import random
from typing import Tuple, Optional, List

import cv2
import numpy as np


# ---------------------------
# Simple image augmentations
# ---------------------------

def add_gaussian_noise(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.01, salt_vs_pepper: float = 0.5) -> np.ndarray:
    if amount <= 0:
        return img
    out = img.copy()
    num_pixels = img.shape[0] * img.shape[1]
    num_salt = int(num_pixels * amount * salt_vs_pepper)
    num_pepper = int(num_pixels * amount * (1.0 - salt_vs_pepper))

    # Salt
    coords = (
        np.random.randint(0, img.shape[0], num_salt),
        np.random.randint(0, img.shape[1], num_salt),
    )
    out[coords] = 255

    # Pepper
    coords = (
        np.random.randint(0, img.shape[0], num_pepper),
        np.random.randint(0, img.shape[1], num_pepper),
    )
    out[coords] = 0
    return out


def motion_blur(img: np.ndarray, ksize: int = 7, angle_deg: Optional[float] = None) -> np.ndarray:
    """Apply directional motion blur to simulate movement/vibration common in 2-wheeler ARAS."""
    ksize = max(3, int(ksize) // 2 * 2 + 1)  # make odd
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()

    if angle_deg is None:
        angle_deg = random.uniform(-20, 20)

    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel = kernel / (kernel.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel)


def add_fog(img: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    """Simple fog/haze effect via lightening + blur. intensity in [0, 0.5]."""
    intensity = max(0.0, min(0.5, float(intensity)))
    h, w = img.shape[:2]
    fog = np.full((h, w, 3), 255, dtype=np.uint8)
    blended = cv2.addWeighted(img, 1.0 - intensity, fog, intensity, 0)
    return cv2.GaussianBlur(blended, (5, 5), 0)


def hsv_color_shift(img: np.ndarray, hue_shift: int = 0, sat_scale: float = 1.0, val_scale: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = (h + hue_shift) % 180
    s = np.clip(s * sat_scale, 0, 255)
    v = np.clip(v * val_scale, 0, 255)
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_rotate(img: np.ndarray, max_deg: int = 10) -> np.ndarray:
    if max_deg <= 0:
        return img
    h, w = img.shape[:2]
    deg = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_brightness_contrast(img: np.ndarray, brightness: float = 0.2, contrast: float = 0.2) -> np.ndarray:
    # brightness [-b, b] * 255, contrast [1-c, 1+c]
    b = random.uniform(-brightness, brightness) * 255.0 if brightness > 0 else 0.0
    c = 1.0 + random.uniform(-contrast, contrast) if contrast > 0 else 1.0
    out = img.astype(np.float32) * c + b
    return np.clip(out, 0, 255).astype(np.uint8)


def random_blur(img: np.ndarray, max_ksize: int = 5) -> np.ndarray:
    if max_ksize < 3:
        return img
    k = random.choice([3, 5]) if max_ksize >= 5 else 3
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_augmentations(
    img: np.ndarray,
    *,
    use_gaussian_noise: bool = True,
    use_salt_pepper: bool = False,
    use_small_rotate: bool = True,
    use_brightness_contrast: bool = True,
    use_gaussian_blur: bool = True,
    use_motion_blur: bool = True,
    use_fog: bool = False,
    use_color_shift: bool = False,
) -> List[np.ndarray]:
    """Create augmented variants tailored for ARAS; no flips (left/right semantics matter)."""
    variants: List[np.ndarray] = []

    if use_gaussian_noise:
        variants.append(add_gaussian_noise(img, sigma=random.choice([10, 15, 25])))
    if use_salt_pepper:
        variants.append(add_salt_pepper_noise(img, amount=random.choice([0.005, 0.01, 0.02])))
    if use_small_rotate:
        variants.append(random_rotate(img, max_deg=8))
    if use_brightness_contrast:
        variants.append(random_brightness_contrast(img, brightness=0.25, contrast=0.25))
    if use_gaussian_blur:
        variants.append(random_blur(img, max_ksize=5))
    if use_motion_blur:
        variants.append(motion_blur(img, ksize=random.choice([5, 7, 9])))
    if use_fog:
        variants.append(add_fog(img, intensity=random.choice([0.1, 0.15, 0.2])))
    if use_color_shift:
        variants.append(hsv_color_shift(img, hue_shift=random.randint(-5, 5), sat_scale=random.uniform(0.9, 1.1), val_scale=random.uniform(0.9, 1.1)))

    return variants


# ---------------------------
# Batch processing utilities
# ---------------------------

def is_image_file(name: str) -> bool:
    name_l = name.lower()
    return any(name_l.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"])


essential_dirs = ["videos", "output_frames", "output_annotation"]


def ensure_dirs(base_dir: str) -> None:
    for d in essential_dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)


def augment_images_in_dir(
    input_dir: str,
    output_dir: Optional[str] = None,
    variants_per_image: int = 2,
    suffix_prefix: str = "aug",
    *,
    use_gaussian_noise: bool = True,
    use_salt_pepper: bool = False,
    use_small_rotate: bool = True,
    use_brightness_contrast: bool = True,
    use_gaussian_blur: bool = True,
    use_motion_blur: bool = True,
    use_fog: bool = False,
    use_color_shift: bool = False,
) -> int:
    """
    Augment all images in input_dir and save alongside originals (or into output_dir if specified).
    - variants_per_image: cap number of variants written per image.
    Returns number of augmented files written.
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    for root, _, files in os.walk(input_dir):
        # Preserve relative structure from input_dir to output_dir
        rel = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel) if rel != "." else output_dir
        os.makedirs(out_root, exist_ok=True)

        for f in files:
            if not is_image_file(f):
                continue
            in_path = os.path.join(root, f)
            img = cv2.imread(in_path)
            if img is None:
                continue

            variants = apply_augmentations(
                img,
                use_gaussian_noise=use_gaussian_noise,
                use_salt_pepper=use_salt_pepper,
                use_small_rotate=use_small_rotate,
                use_brightness_contrast=use_brightness_contrast,
                use_gaussian_blur=use_gaussian_blur,
                use_motion_blur=use_motion_blur,
                use_fog=use_fog,
                use_color_shift=use_color_shift,
            )
            random.shuffle(variants)
            for i, aug in enumerate(variants[:max(0, int(variants_per_image))]):
                name, ext = os.path.splitext(f)
                out_path = os.path.join(out_root, f"{name}_{suffix_prefix}{i+1}{ext}")
                cv2.imwrite(out_path, aug)
                written += 1

    return written


def extract_frames_every(video_path: str, output_dir: str, interval_seconds: int = 3) -> int:
    """
    Extract one frame every `interval_seconds` from video into output_dir.
    Returns number of frames written.
    """
    os.makedirs(output_dir, exist_ok=True)
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        return 0

    fps = cam.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_secs = total_frames / max(fps, 1.0)

    frame_index = 0
    current_time = 0.0
    written = 0

    while current_time < duration_secs:
        cam.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cam.read()
        if not ret or frame is None:
            break
        filename = os.path.join(output_dir, f"frame{frame_index}.jpg")
        cv2.imwrite(filename, frame)
        written += 1
        frame_index += 1
        current_time += max(1, int(interval_seconds))

    cam.release()
    return written
