import os
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, label
from skimage import measure

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
MIN_CLUSTER_SIZE = 150


def _load_volume(folder):
    slices = []
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith(VALID_EXTENSIONS):
            img = cv2.imread(os.path.join(folder, file), 0)
            if img is not None:
                slices.append(cv2.resize(img, (256, 256)))
    if not slices:
        raise ValueError("No scan slices found for progression analysis.")
    volume = np.stack(slices, axis=0).astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    return gaussian_filter(volume, sigma=1)


def _segment(volume):
    organ_mask = volume > np.percentile(volume, 45)
    lesion_thresh = np.percentile(volume[organ_mask], 98) if np.any(organ_mask) else 1.0
    lesion_raw = (volume > lesion_thresh) & organ_mask
    labeled, n = label(lesion_raw)
    lesion_mask = np.zeros_like(lesion_raw)
    for i in range(1, n + 1):
        if np.sum(labeled == i) >= MIN_CLUSTER_SIZE:
            lesion_mask[labeled == i] = 1
    return organ_mask, lesion_mask


def _match_depth(mask1, mask2):
    d = min(mask1.shape[0], mask2.shape[0])
    return mask1[:d], mask2[:d]


def _safe_mesh(fig, mask, color, opacity, name):
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
            lighting=dict(ambient=0.35, diffuse=0.8, roughness=0.35, specular=0.5),
        )
    )


def compare_progression(folder1, folder2):
    vol1 = _load_volume(folder1)
    vol2 = _load_volume(folder2)

    organ1, lesion1 = _segment(vol1)
    organ2, lesion2 = _segment(vol2)
    organ1, organ2 = _match_depth(organ1, organ2)
    lesion1, lesion2 = _match_depth(lesion1, lesion2)

    growth = lesion2 & (~lesion1)
    regression = lesion1 & (~lesion2)

    o1 = float(np.sum(organ1))
    o2 = float(np.sum(organ2))
    l1 = float(np.sum(lesion1))
    l2 = float(np.sum(lesion2))
    t1_burden = (l1 / o1) * 100 if o1 else 0.0
    t2_burden = (l2 / o2) * 100 if o2 else 0.0
    absolute_change = t2_burden - t1_burden
    relative_change = (absolute_change / t1_burden) * 100 if t1_burden > 0 else 0.0

    fig = go.Figure()
    _safe_mesh(fig, organ2, "lightpink", 0.22, "Heart Tissue (T2)")
    _safe_mesh(fig, lesion1, "orange", 0.35, "Lesion at T1")
    _safe_mesh(fig, lesion2, "red", 0.72, "Lesion at T2")
    _safe_mesh(fig, growth, "limegreen", 0.9, "New Growth")
    _safe_mesh(fig, regression, "deepskyblue", 0.82, "Regression")

    fig.update_layout(
        title="Heart Progression (T1 vs T2)",
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(eye=dict(x=1.45, y=1.3, z=1.2)),
            bgcolor="rgb(255,247,249)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig, {
        "t1_burden": t1_burden,
        "t2_burden": t2_burden,
        "absolute_change": absolute_change,
        "relative_change": relative_change,
        "growth_voxels": int(np.sum(growth)),
        "regression_voxels": int(np.sum(regression)),
    }
