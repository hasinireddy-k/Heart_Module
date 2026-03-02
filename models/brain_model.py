import os
import cv2
import numpy as np
from reconstruction.brain_3d import load_brain_volume, segment_brain_and_tumor, compute_tumor_percentage

MODEL_PATH = os.path.join("models", "brain_model.h5")
IMG_SIZE = 128
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
FALLBACK_MODEL_PATH = os.path.join("brain_module", "models", "brain_model.h5")
MAX_SLICES_FOR_INFERENCE = 24


def _iter_image_paths(folder_path):
    paths = []
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(VALID_EXTENSIONS):
            paths.append(os.path.join(folder_path, name))
    return paths


def _sample_paths_evenly(paths, max_count):
    if len(paths) <= max_count:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_count, dtype=int)
    return [paths[i] for i in indices]


def _load_model():
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    if not os.path.exists(model_path):
        return None

    return load_model(model_path)


def _preprocess_gray(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE improves local contrast in MR-like grayscale images.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    return img


def _tta_predict(model, gray_img):
    variants = [
        gray_img,
        np.fliplr(gray_img),
        np.flipud(gray_img),
    ]
    batch = np.stack(variants, axis=0).reshape(len(variants), IMG_SIZE, IMG_SIZE, 1)
    preds = model.predict(batch, verbose=0).reshape(-1)
    return float(np.mean(preds))


def _predict_multislice(folder_path, model):
    image_paths = _iter_image_paths(folder_path)
    if not image_paths:
        raise ValueError("No valid brain image found for prediction.")

    sampled_paths = _sample_paths_evenly(image_paths, MAX_SLICES_FOR_INFERENCE)
    slice_scores = []
    for image_path in sampled_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = _preprocess_gray(img)
        score = _tta_predict(model, gray)
        slice_scores.append(score)

    if not slice_scores:
        raise ValueError("Unable to read uploaded brain image.")

    # Robust aggregation: blend median and high-risk quantile across slices.
    scores = np.array(slice_scores, dtype=np.float32)
    return float(0.6 * np.median(scores) + 0.4 * np.percentile(scores, 80))


def _heuristic_prediction(folder_path):
    volume = load_brain_volume(folder_path)
    brain_mask, tumor_mask = segment_brain_and_tumor(volume)
    tumor_percent = compute_tumor_percentage(brain_mask, tumor_mask)

    if tumor_percent >= 1.0:
        confidence = min(99.0, 55.0 + tumor_percent * 4.2)
        return "Tumor Detected", confidence, "Segmentation Heuristic"
    confidence = max(55.0, 96.0 - (tumor_percent * 30.0))
    return "No Tumor", confidence, "Segmentation Heuristic"


def predict_brain_tumor(folder_path):
    model = _load_model()
    if model is None:
        return _heuristic_prediction(folder_path)

    prediction = _predict_multislice(folder_path, model)

    # Blend CNN output with 3D segmentation burden for more stable decisions.
    volume = load_brain_volume(folder_path)
    brain_mask, tumor_mask = segment_brain_and_tumor(volume)
    tumor_percent = compute_tumor_percentage(brain_mask, tumor_mask)
    burden_prior = np.clip(tumor_percent / 6.0, 0.0, 1.0)
    final_score = float(0.85 * prediction + 0.15 * burden_prior)

    if final_score > 0.5:
        return "Tumor Detected", final_score * 100, "Pretrained CNN + TTA + Multi-slice"
    return "No Tumor", (1 - final_score) * 100, "Pretrained CNN + TTA + Multi-slice"
