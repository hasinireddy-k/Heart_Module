import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_opening, binary_fill_holes
from skimage import measure
import plotly.graph_objects as go

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
MIN_TUMOR_CLUSTER_SIZE = 120


def load_brain_volume(slice_folder):
    slices = []
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

    for file in sorted(os.listdir(slice_folder)):
        if file.lower().endswith(VALID_EXTENSIONS):
            img = cv2.imread(os.path.join(slice_folder, file), 0)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = clahe.apply(img)
                slices.append(img)

    if len(slices) == 0:
        raise ValueError("No brain slice images found.")

    volume = np.stack(slices, axis=0).astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    return gaussian_filter(volume, sigma=(0.8, 1.0, 1.0))


def segment_brain_and_tumor(volume):
    brain_mask = volume > np.percentile(volume, 43)
    brain_mask = binary_closing(brain_mask, iterations=1)
    brain_mask = binary_opening(brain_mask, iterations=1)
    brain_mask = binary_fill_holes(brain_mask)

    tumor_threshold = np.percentile(volume[brain_mask], 97.9) if np.any(brain_mask) else 0.0
    raw_tumor = (volume > tumor_threshold) & brain_mask
    raw_tumor = binary_opening(raw_tumor, iterations=1)
    raw_tumor = binary_closing(raw_tumor, iterations=1)

    labeled, num_features = label(raw_tumor)
    tumor_mask = np.zeros_like(raw_tumor)
    for region in range(1, num_features + 1):
        if np.sum(labeled == region) >= MIN_TUMOR_CLUSTER_SIZE:
            tumor_mask[labeled == region] = 1

    return brain_mask, tumor_mask


def compute_tumor_percentage(brain_mask, tumor_mask):
    brain_volume = float(np.sum(brain_mask))
    tumor_volume = float(np.sum(tumor_mask))
    return (tumor_volume / brain_volume) * 100 if brain_volume else 0.0


def _add_mesh(fig, mask, color, opacity, name):
    if not np.any(mask):
        return
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0)
    except ValueError:
        return
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            lighting=dict(ambient=0.32, diffuse=0.86, roughness=0.3, specular=0.58, fresnel=0.12),
            lightposition=dict(x=125, y=110, z=180),
        )
    )


def generate_3d_brain(slice_folder):
    volume = load_brain_volume(slice_folder)
    brain_mask, tumor_mask = segment_brain_and_tumor(volume)
    tumor_percent = compute_tumor_percentage(brain_mask, tumor_mask)
    tumor_centroid = None
    if np.any(tumor_mask):
        coords = np.argwhere(tumor_mask > 0)
        centroid = np.mean(coords, axis=0)
        tumor_centroid = {
            "z": round(float(centroid[0]), 1),
            "y": round(float(centroid[1]), 1),
            "x": round(float(centroid[2]), 1),
        }

    fig = go.Figure()
    _add_mesh(fig, brain_mask, "lightsteelblue", 0.33, "Brain Tissue")
    _add_mesh(fig, tumor_mask, "crimson", 0.92, "Tumor Candidate")

    fig.update_layout(
        title=f"3D Brain Reconstruction (Tumor Burden: {tumor_percent:.2f}%)",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=1.6, y=1.46, z=1.26)),
            bgcolor="rgb(243,248,255)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig, tumor_percent, tumor_centroid
