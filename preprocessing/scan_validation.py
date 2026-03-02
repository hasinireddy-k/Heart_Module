import os
import cv2
import numpy as np

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
MISMATCH_BLOCK_CONFIDENCE = 0.9


def _iter_image_paths(folder_path):
    paths = []
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(VALID_EXTENSIONS):
            paths.append(os.path.join(folder_path, name))
    return paths


def _sample_even(paths, max_count=16):
    if len(paths) <= max_count:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_count, dtype=int)
    return [paths[i] for i in idx]


def _safe_corr(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))


def _largest_component(mask):
    mask_uint = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _circularity(contour):
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri <= 1e-6:
        return 0.0
    return float(np.clip((4.0 * np.pi * area) / (peri * peri), 0.0, 1.0))


def _brain_heart_scores(gray):
    gray = cv2.resize(gray, (256, 256))
    img = gray.astype(np.float32) / 255.0

    h, w = img.shape
    m0, m1 = int(h * 0.2), int(h * 0.8)
    n0, n1 = int(w * 0.2), int(w * 0.8)
    mid = img[m0:m1, n0:n1]

    flipped = np.fliplr(mid)
    symmetry = (_safe_corr(mid, flipped) + 1.0) / 2.0

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mid = otsu[m0:m1, n0:n1] > 0
    contour = _largest_component(otsu_mid)
    circ = _circularity(contour) if contour is not None else 0.0

    ring = np.zeros_like(img, dtype=np.uint8)
    bw = 14
    ring[:bw, :] = 1
    ring[-bw:, :] = 1
    ring[:, :bw] = 1
    ring[:, -bw:] = 1
    edge_bright = float(np.mean(img[ring == 1] > 0.62))

    dark_mask = (mid < 0.18).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    large_dark = 0
    min_area = int(0.01 * dark_mask.size)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_dark += 1
    lung_like = 1.0 if large_dark >= 2 else 0.0

    brain_score = 0.55 * symmetry + 0.35 * circ + 0.1 * edge_bright
    heart_score = (
        0.62 * (1.0 - symmetry)
        + 0.38 * (1.0 - circ)
        + 0.12 * (1.0 - edge_bright)
        + 0.05 * lung_like
    )
    return float(brain_score), float(heart_score)


def detect_scan_type(folder_path):
    paths = _iter_image_paths(folder_path)
    if not paths:
        return "unknown", 0.0

    brain_scores = []
    heart_scores = []
    for p in _sample_even(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        b, h = _brain_heart_scores(img)
        brain_scores.append(b)
        heart_scores.append(h)

    if not brain_scores:
        return "unknown", 0.0

    b_score = float(np.median(brain_scores))
    h_score = float(np.median(heart_scores))
    margin = abs(b_score - h_score)
    confidence = float(np.clip(0.5 + margin, 0.0, 1.0))

    if b_score >= 0.5 and b_score >= (h_score - 0.05):
        return "brain", confidence
    if h_score >= 0.7 and h_score >= (b_score - 0.05):
        return "heart", confidence

    if margin < 0.01:
        return "unknown", confidence
    return ("brain", confidence) if b_score > h_score else ("heart", confidence)


def validate_scan_folder(folder_path, expected):
    predicted, confidence = detect_scan_type(folder_path)
    if predicted == expected:
        return True, ""
    if predicted == "unknown":
        # Avoid blocking borderline/low-quality studies. We only hard-block
        # when the uploaded scans are confidently detected as the wrong organ.
        return True, ""
    if confidence < MISMATCH_BLOCK_CONFIDENCE:
        return True, ""
    return False, (
        f"Scan type mismatch: expected {expected} scans, but detected {predicted} scans "
        f"(confidence {confidence * 100:.1f}%)."
    )
