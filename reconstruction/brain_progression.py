import numpy as np
import plotly.graph_objects as go
from reconstruction.brain_3d import (
    load_brain_volume,
    segment_brain_and_tumor,
    compute_tumor_percentage,
    _add_mesh,
)


def _match_depth(mask1, mask2):
    depth = min(mask1.shape[0], mask2.shape[0])
    return mask1[:depth], mask2[:depth]


def compare_brain_progression(folder_t1, folder_t2):
    vol1 = load_brain_volume(folder_t1)
    vol2 = load_brain_volume(folder_t2)

    brain1, tumor1 = segment_brain_and_tumor(vol1)
    brain2, tumor2 = segment_brain_and_tumor(vol2)
    brain1, brain2 = _match_depth(brain1, brain2)
    tumor1, tumor2 = _match_depth(tumor1, tumor2)

    growth_mask = tumor2 & (~tumor1)
    regression_mask = tumor1 & (~tumor2)

    t1_burden = compute_tumor_percentage(brain1, tumor1)
    t2_burden = compute_tumor_percentage(brain2, tumor2)
    absolute_growth = t2_burden - t1_burden
    relative_growth = (absolute_growth / t1_burden) * 100 if t1_burden > 0 else 0.0

    fig = go.Figure()
    _add_mesh(fig, brain2, "lightsteelblue", 0.22, "Brain Tissue (T2)")
    _add_mesh(fig, tumor1, "goldenrod", 0.35, "Tumor at T1")
    _add_mesh(fig, tumor2, "crimson", 0.7, "Tumor at T2")
    _add_mesh(fig, growth_mask, "limegreen", 0.9, "New Growth")
    _add_mesh(fig, regression_mask, "deepskyblue", 0.85, "Regression")

    fig.update_layout(
        title="Brain Tumor Progression (T1 vs T2)",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=1.5, y=1.4, z=1.2)),
            bgcolor="rgb(247,249,252)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig, {
        "t1_burden": float(t1_burden),
        "t2_burden": float(t2_burden),
        "absolute_growth": float(absolute_growth),
        "relative_growth": float(relative_growth),
        "growth_voxels": int(np.sum(growth_mask)),
        "regression_voxels": int(np.sum(regression_mask)),
    }
