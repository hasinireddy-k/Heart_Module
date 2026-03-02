import os
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_opening, binary_fill_holes
from skimage import measure

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
MIN_CLUSTER_SIZE = 180


def _load_heart_volume(slice_folder):
    slices = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for file in sorted(os.listdir(slice_folder)):
        if file.lower().endswith(VALID_EXTENSIONS):
            img = cv2.imread(os.path.join(slice_folder, file), 0)
            if img is not None:
                resized = cv2.resize(img, (256, 256))
                denoised = cv2.GaussianBlur(resized, (3, 3), 0)
                enhanced = clahe.apply(denoised)
                slices.append(enhanced)
    if not slices:
        raise ValueError("No slice images found.")
    volume = np.stack(slices, axis=0).astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    return gaussian_filter(volume, sigma=(0.8, 1.0, 1.0))


def _segment_heart_and_lesion(volume):
    base_thresh = np.percentile(volume, 43)
    heart_mask = volume > base_thresh
    heart_mask = binary_closing(heart_mask, iterations=1)
    heart_mask = binary_opening(heart_mask, iterations=1)
    heart_mask = binary_fill_holes(heart_mask)

    lesion_thresh = np.percentile(volume[heart_mask], 97.8) if np.any(heart_mask) else 1.0
    lesion_raw = (volume > lesion_thresh) & heart_mask
    lesion_raw = binary_opening(lesion_raw, iterations=1)
    lesion_raw = binary_closing(lesion_raw, iterations=1)
    labeled, n = label(lesion_raw)
    lesion_mask = np.zeros_like(lesion_raw)
    for i in range(1, n + 1):
        if np.sum(labeled == i) >= MIN_CLUSTER_SIZE:
            lesion_mask[labeled == i] = 1
    return heart_mask, lesion_mask


def _add_mesh(fig, mask, color, opacity, name):
    if not np.any(mask):
        return
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0)
    except ValueError:
        return
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, name=name,
            lighting=dict(ambient=0.32, diffuse=0.86, roughness=0.28, specular=0.62, fresnel=0.14),
            lightposition=dict(x=110, y=130, z=190),
        )
    )


def generate_3d_heart(slice_folder):
    volume = _load_heart_volume(slice_folder)
    heart_mask, lesion_mask = _segment_heart_and_lesion(volume)

    heart_volume = float(np.sum(heart_mask))
    lesion_volume = float(np.sum(lesion_mask))
    tumor_percentage = (lesion_volume / heart_volume) * 100 if heart_volume else 0.0
    lesion_centroid = None
    if lesion_volume > 0:
        coords = np.argwhere(lesion_mask > 0)
        centroid = np.mean(coords, axis=0)
        lesion_centroid = {
            "z": round(float(centroid[0]), 1),
            "y": round(float(centroid[1]), 1),
            "x": round(float(centroid[2]), 1),
        }

    if tumor_percentage < 2:
        severity = "Mild"
    elif tumor_percentage < 6:
        severity = "Moderate"
    else:
        severity = "Severe"

    fig = go.Figure()
    _add_mesh(fig, heart_mask, "lightpink", 0.34, "Heart Tissue")
    _add_mesh(fig, lesion_mask, "crimson", 0.93, "Abnormal Growth")

    fig.update_layout(
        title=f"Advanced 3D Heart Reconstruction (Lesion Burden: {tumor_percentage:.2f}%)",
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(eye=dict(x=1.55, y=1.34, z=1.24)),
            bgcolor="rgb(245,250,255)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig, tumor_percentage, severity, lesion_centroid
