import os
import cv2
import numpy as np

MODEL_CANDIDATES = [
    os.path.join("models", "lung_best_model.keras"),
    os.path.join("models", "lung_best_model.h5"),
    os.path.join("models", "lung_model.keras"),
    os.path.join("models", "lung_model.h5"),
    os.path.join("brain_module", "models", "lung_best_model.keras"),
    os.path.join("brain_module", "models", "lung_model.h5"),
]
IMG_SIZE = 224
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_SLICES_FOR_INFERENCE = 24
DECISION_THRESHOLD = 0.5


def _iter_image_paths(folder_path):
    paths = []
    if not os.path.isdir(folder_path):
        return paths
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(VALID_EXTENSIONS):
            paths.append(os.path.join(folder_path, name))
    return paths


def _sample_paths_evenly(paths, max_count):
    if len(paths) <= max_count:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_count, dtype=int)
    return [paths[i] for i in indices]


def _load_custom_model():
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None

    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            return load_model(model_path)
    return None


def _load_pretrained_backbone():
    try:
        from tensorflow.keras.applications import EfficientNetB0
    except ImportError:
        return None

    try:
        return EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    except Exception:
        return None


def _prepare_rgb(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def _slice_lung_score(gray):
    gray = cv2.resize(gray, (256, 256))
    gray = gray.astype(np.float32) / 255.0

    mid = gray[48:208, 48:208]
    symmetry = np.corrcoef(mid.flatten(), np.fliplr(mid).flatten())[0, 1]
    if np.isnan(symmetry):
        symmetry = 0.0
    symmetry = float(np.clip((symmetry + 1.0) / 2.0, 0.0, 1.0))

    bright_focus = float(np.mean(mid > 0.82))
    texture = float(np.std(mid))
    score = 0.55 * bright_focus + 0.30 * texture + 0.15 * (1.0 - symmetry)
    return float(np.clip(score, 0.0, 1.0))


def _heuristic_prediction(folder_path):
    paths = _iter_image_paths(folder_path)
    if not paths:
        raise ValueError("No valid lung image found. Upload PNG/JPG slices.")

    sampled = _sample_paths_evenly(paths, MAX_SLICES_FOR_INFERENCE)
    scores = []
    for image_path in sampled:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        scores.append(_slice_lung_score(img))

    if not scores:
        raise ValueError("Unable to read uploaded lung images.")

    risk = float(np.percentile(np.array(scores, dtype=np.float32), 70) * 100.0)
    if risk >= 52.0:
        confidence = float(min(98.0, 55.0 + (risk - 50.0) * 1.6))
        return "Potential Pulmonary Abnormality", confidence, "Radiology Heuristic", risk
    confidence = float(min(98.0, 55.0 + (50.0 - risk) * 1.6))
    return "Likely Normal", confidence, "Radiology Heuristic", risk


def _predict_with_custom_model(folder_path, model):
    paths = _iter_image_paths(folder_path)
    if not paths:
        raise ValueError("No valid lung image found. Upload PNG/JPG slices.")

    sampled = _sample_paths_evenly(paths, MAX_SLICES_FOR_INFERENCE)
    preds = []
    for image_path in sampled:
        img = cv2.imread(image_path)
        if img is None:
            continue
        base = _prepare_rgb(img) / 255.0
        variants = [
            base,
            np.fliplr(base),
            np.flipud(base),
        ]
        x = np.stack(variants, axis=0)
        pred = float(np.mean(model.predict(x, verbose=0).reshape(-1)))
        preds.append(pred)

    if not preds:
        raise ValueError("Unable to read uploaded lung images.")
    scores = np.array(preds, dtype=np.float32)
    return float(0.65 * np.median(scores) + 0.35 * np.percentile(scores, 75)), float(np.std(scores))


def _predict_with_pretrained(folder_path, backbone):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    paths = _iter_image_paths(folder_path)
    if not paths:
        raise ValueError("No valid lung image found. Upload PNG/JPG slices.")

    sampled = _sample_paths_evenly(paths, MAX_SLICES_FOR_INFERENCE)
    batch = []
    for image_path in sampled:
        img = cv2.imread(image_path)
        if img is None:
            continue
        batch.append(_prepare_rgb(img))

    if not batch:
        raise ValueError("Unable to read uploaded lung images.")

    x = preprocess_input(np.stack(batch, axis=0))
    features = backbone.predict(x, verbose=0)
    activation = np.mean(np.abs(features), axis=1)
    z = (activation - np.mean(activation)) / (np.std(activation) + 1e-6)
    score = float(np.clip(0.5 + 0.22 * np.mean(z), 0.0, 1.0))
    return score, float(np.std(activation))


def _calibrate_confidence(score, spread):
    margin = abs(score - DECISION_THRESHOLD) * 2.0
    stability = float(np.clip(1.0 - min(1.0, spread), 0.0, 1.0))
    calibrated = float(np.clip(0.55 + 0.40 * margin * stability, 0.52, 0.99))
    return calibrated * 100.0


def predict_lung_condition(folder_path):
    custom_model = _load_custom_model()
    if custom_model is not None:
        model_score, spread = _predict_with_custom_model(folder_path, custom_model)
        method = "Custom Pretrained Lung CNN"
    else:
        backbone = _load_pretrained_backbone()
        if backbone is None:
            return _heuristic_prediction(folder_path)
        model_score, spread = _predict_with_pretrained(folder_path, backbone)
        method = "EfficientNetB0 ImageNet Transfer Inference"

    heuristic_label, _, _, heuristic_risk = _heuristic_prediction(folder_path)
    heuristic_score = heuristic_risk / 100.0
    score = float(np.clip(0.85 * model_score + 0.15 * heuristic_score, 0.0, 1.0))
    risk = float(score * 100.0)
    confidence = _calibrate_confidence(score, spread)
    if score >= DECISION_THRESHOLD:
        return "Potential Pulmonary Abnormality", confidence, method, risk
    if heuristic_label == "Potential Pulmonary Abnormality" and score >= 0.47:
        return "Potential Pulmonary Abnormality", confidence, method + " + Heuristic Prior", risk
    return "Likely Normal", confidence, method, risk
