import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import gaussian_filter
import os
import cv2

def load_volume(folder):

    slices = []

    for file in sorted(os.listdir(folder)):
        if file.endswith(".png") or file.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, file), 0)
            img = cv2.resize(img, (256,256))
            slices.append(img)

    volume = np.stack(slices, axis=0)
    volume = gaussian_filter(volume, sigma=1)
    volume = volume > np.mean(volume)

    return volume


def compare_progression(folder1, folder2):

    vol1 = load_volume(folder1)
    vol2 = load_volume(folder2)

    diff = vol2.astype(int) - vol1.astype(int)

    verts, faces, _, _ = measure.marching_cubes(diff, level=0)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:,0],
            y=verts[:,1],
            z=verts[:,2],
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            color='yellow',
            opacity=0.7
        )
    ])

    fig.update_layout(title="Disease Progression Visualization")

    return fig
