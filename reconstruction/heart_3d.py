import os
import cv2
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter, label
import plotly.graph_objects as go

def generate_3d_heart(slice_folder):

    slices = []

    for file in sorted(os.listdir(slice_folder)):
        if file.endswith(".png") or file.endswith(".jpg"):
            img = cv2.imread(os.path.join(slice_folder, file), 0)

            if img is not None:
                img = cv2.resize(img, (256,256))
                slices.append(img)

    if len(slices) == 0:
        raise ValueError("No slice images found!")

    volume = np.stack(slices, axis=0).astype(np.float32)

    # Normalize intensity
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Smooth volume
    volume = gaussian_filter(volume, sigma=1)

    # Heart mask
    heart_mask = volume > 0.3

    # Tumor detection using top 2% brightest voxels
    threshold = np.percentile(volume, 98)
    tumor_mask = volume > threshold

    # Remove small noise clusters
    labeled_array, num_features = label(tumor_mask)
    cleaned_tumor = np.zeros_like(tumor_mask)

    for region in range(1, num_features + 1):
        region_size = np.sum(labeled_array == region)
        if region_size > 800:
            cleaned_tumor[labeled_array == region] = 1

    # Calculate volumes
    heart_volume = np.sum(heart_mask)
    tumor_volume = np.sum(cleaned_tumor)

    tumor_percentage = (tumor_volume / heart_volume) * 100 if heart_volume != 0 else 0

    # Severity classification
    if tumor_percentage < 2:
        severity = "Mild"
    elif tumor_percentage < 6:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Create 3D meshes
    verts1, faces1, _, _ = measure.marching_cubes(heart_mask, level=0)

    fig = go.Figure()

    # Heart
    fig.add_trace(go.Mesh3d(
        x=verts1[:,0],
        y=verts1[:,1],
        z=verts1[:,2],
        i=faces1[:,0],
        j=faces1[:,1],
        k=faces1[:,2],
        color='lightpink',
        opacity=0.35,
        name="Heart Tissue"
    ))

    # Tumor if exists
    if tumor_volume > 0:
        verts2, faces2, _, _ = measure.marching_cubes(cleaned_tumor, level=0)

        fig.add_trace(go.Mesh3d(
            x=verts2[:,0],
            y=verts2[:,1],
            z=verts2[:,2],
            i=faces2[:,0],
            j=faces2[:,1],
            k=faces2[:,2],
            color='red',
            opacity=0.95,
            name="Abnormal Growth"
        ))

    fig.update_layout(
        title="3D Heart Reconstruction with Tumor Detection",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False
        )
    )

    return fig, tumor_percentage, severity
