# -*- coding: utf-8 -*-
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication-size fonts.
CELL_VALUE_FONTSIZE = 20          # numbers inside heatmap cells
BASE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 27
TITLE_FONTSIZE = 23
XTICK_FONTSIZE = 21
YTICK_FONTSIZE = 23
COLORBAR_TICK_FONTSIZE = 23

PROPOSED_DETECTOR = 'HAST'
# Order of values stored in the dictionaries: [HAST, Feature-KS, OCDD, ADWIN, EDDM, DDM]
DATA_ORDER = [PROPOSED_DETECTOR, 'Feature-KS', 'OCDD', 'ADWIN', 'EDDM', 'DDM']

# Order displayed in heatmaps: [HAST, OCDD, Feature-KS, ADWIN, DDM, EDDM]
detectors = [PROPOSED_DETECTOR, 'OCDD', 'Feature-KS', 'ADWIN', 'DDM', 'EDDM']
detector_col_idx = [DATA_ORDER.index(det) for det in detectors]

drifts = [3, 5, 10, 15]
features = [30, 60, 90]

D1_gradual = {
    (3, 30): [5.07, 7.853333333333333, 15.35, 17.67, 3.02, 17.17],
    (3, 60): [6.43, 8.0, 17.08, 21.19, 3.00, 10.92],
    (3, 90): [3.61, 8.0, 16.09, 16.37, 3.90, 8.18],
    (5, 30): [2.76, 7.911111111111111, 10.11, 11.71, 2.95, 9.76],
    (5, 60): [3.42, 8.0, 10.51, 11.34, 3.56, 4.68],
    (5, 90): [4.51, 8.0, 10.05, 9.24, 3.45, 4.76],
    (10, 30): [1.54, 4.814285714285714, 4.74, 6.30, 3.14, 0.72],
    (10, 60): [2.58, 5.099523809523809, 4.52, 5.76, 2.80, 3.11],
    (10, 90): [2.68, 5.404285714285714, 4.78, 5.82, 3.65, 2.15],
    (15, 30): [0.89, 3.092916666666672, 4.10, 4.13, 2.61, 3.31],
    (15, 60): [1.82, 4.321388888888892, 3.84, 3.94, 2.27, 2.42],
    (15, 90): [0.60, 4.584861111111113, 4.20, 4.30, 2.52, 1.96],
}

D2_gradual = {
    (3, 30): [3.39, 2.077777777777772, 3.82, 28.72, 16.04, 1.33],
    (3, 60): [5.60, 2.333333333333326, 6.49, 41.06, 10.53, 1.18],
    (3, 90): [13.63, 2.699999999999993, 4.48, 31.89, 14.06, 1.71],
    (5, 30): [2.28, 1.48, 5.92, 31.26, 22.78, 0.90],
    (5, 60): [7.88, 1.6800000000000002, 8.06, 43.34, 26.62, 0.74],
    (5, 90): [12.00, 1.94, 5.06, 35.48, 16.22, 1.32],
    (10, 30): [1.80, 2.8999999999999995, 7.78, 34.47, 20.95, 0.95],
    (10, 60): [6.09, 3.14, 7.83, 28.60, 23.67, 1.17],
    (10, 90): [6.42, 3.39, 9.68, 29.00, 11.13, 1.10],
    (15, 30): [1.05, 2.997777777777784, 7.56, 40.32, 19.38, 0.90],
    (15, 60): [3.30, 4.131111111111115, 11.03, 26.51, 19.25, 1.03],
    (15, 90): [4.21, 4.380000000000003, 15.57, 25.91, 38.51, 0.77],
}

R_gradual = {
    (3, 30): [0.15, 0.49, 0.49, 0.88, 0.41, 0.90],
    (3, 60): [0.05, 0.50, 0.43, 0.97, 0.26, 0.77],
    (3, 90): [0.10, 0.50, 0.46, 0.85, 0.23, 0.46],
    (5, 30): [0.07, 0.4944444444444444, 0.36, 0.79, 0.59, 0.85],
    (5, 60): [0.12, 0.50, 0.33, 0.71, 0.82, 0.67],
    (5, 90): [0.23, 0.50, 0.36, 0.73, 0.22, 0.59],
    (10, 30): [0.05, 0.2857142857142857, 0.12, 0.58, 0.81, 0.74],
    (10, 60): [0.18, 0.3047619047619047, 0.12, 0.56, 1.50, 0.58],
    (10, 90): [0.22, 0.3238095238095239, 0.11, 0.57, 0.44, 0.40],
    (15, 30): [0.02, 0.025, 0.40, 0.42, 1.62, 0.67],
    (15, 60): [0.10, 0.05, 0.80, 1.05, 1.64, 0.48],
    (15, 90): [0.18, 0.05625, 1.02, 0.60, 2.09, 0.08],
}

D1_abrupt = {
    (3, 30): [2.66, 3.5427777777777743, 15.09, 20.53, 1.87, 19.96],
    (3, 60): [3.05, 4.083333333333329, 13.89, 21.96, 2.43, 18.57],
    (3, 90): [2.29, 4.546666666666662, 16.73, 19.26, 2.39, 13.61],
    (5, 30): [1.55, 0.0, 8.74, 12.06, 1.83, 11.32],
    (5, 60): [2.37, 0.0, 9.69, 11.79, 2.35, 9.93],
    (5, 90): [2.23, 0.0, 9.76, 12.16, 2.09, 7.70],
    (10, 30): [0.81, 5.357142857142857, 4.15, 6.35, 2.42, 5.69],
    (10, 60): [1.37, 5.5, 4.24, 6.36, 2.21, 4.69],
    (10, 90): [0.92, 5.5, 4.13, 5.83, 2.70, 3.94],
    (15, 30): [0.36, 2.173611111111115, 3.48, 4.04, 2.04, 3.69],
    (15, 60): [0.90, 2.0918055555555592, 3.29, 4.22, 2.05, 3.20],
    (15, 90): [0.79, 2.0373611111111165, 3.73, 4.14, 2.14, 2.42],
}

D2_abrupt = {
    (3, 30): [0.30, 0.2999999999999933, 3.18, 29.63, 7.24, 0.51],
    (3, 60): [1.28, 0.3333333333333262, 2.50, 46.33, 2.33, 0.39],
    (3, 90): [8.68, 0.3333333333333262, 3.99, 32.41, 7.72, 0.39],
    (5, 30): [0.28, 0.0, 3.30, 27.38, 10.28, 0.32],
    (5, 60): [3.78, 0.0, 2.64, 39.00, 12.10, 0.12],
    (5, 90): [5.94, 0.0, 3.18, 28.74, 8.70, 0.04],
    (10, 30): [1.10, 3.38, 5.38, 36.78, 19.61, 0.66],
    (10, 60): [2.78, 3.5, 5.71, 33.01, 13.66, 0.57],
    (10, 90): [6.09, 3.5, 5.21, 19.66, 17.36, 0.53],
    (15, 30): [0.44, 2.100000000000004, 6.02, 36.93, 17.94, 0.59],
    (15, 60): [2.25, 2.020000000000004, 8.79, 27.89, 19.57, 0.42],
    (15, 90): [3.08, 1.9888888888888947, 13.29, 27.45, 30.40, 0.37],
}

R_abrupt = {
    (3, 30): [0.17, 0.215, 0.55, 0.85, 0.12, 0.93],
    (3, 60): [0.12, 0.25, 0.57, 0.94, 0.05, 0.89],
    (3, 90): [0.07, 0.28, 0.54, 0.79, 0.15, 0.67],
    (5, 30): [0.10, 0.0, 0.46, 0.76, 0.17, 0.89],
    (5, 60): [0.10, 0.0, 0.47, 1.07, 0.47, 0.83],
    (5, 90): [0.13, 0.0, 0.47, 0.76, 0.21, 0.65],
    (10, 30): [0.08, 0.3238095238095239, 0.17, 0.59, 0.95, 0.81],
    (10, 60): [0.09, 0.3333333333333333, 0.21, 0.63, 0.56, 0.69],
    (10, 90): [0.24, 0.3333333333333333, 0.20, 0.58, 0.69, 0.55],
    (15, 30): [0.05, 0.01875, 0.19, 0.49, 1.35, 0.73],
    (15, 60): [0.08, 0.01875, 0.46, 0.34, 1.28, 0.52],
    (15, 90): [0.12, 0.0125, 0.58, 0.49, 1.51, 0.44],
}

metric_data = {
    ('abrupt', 'D1'): D1_abrupt,
    ('abrupt', 'D2'): D2_abrupt,
    ('abrupt', 'R'): R_abrupt,
    ('gradual', 'D1'): D1_gradual,
    ('gradual', 'D2'): D2_gradual,
    ('gradual', 'R'): R_gradual,
}

metric_labels = {
    'D1': r'$D1$ score',
    'D2': r'$D2$ score',
    'R': r'$R$ score',
}

row_labels = [f'{d}d {f}f' for d in drifts for f in features]

# Build CSV with all source data.
records = []
for (drift_type, metric), values in metric_data.items():
    for (n_drifts, n_features), vals in values.items():
        if len(vals) != len(detectors):
            raise ValueError(
                f'Wrong number of values for {drift_type}, {metric}, '
                f'{n_drifts}d {n_features}f: expected {len(detectors)}, got {len(vals)}'
            )
        for det, idx in zip(detectors, detector_col_idx):
            val = vals[idx]
            records.append({
                'DriftType': drift_type,
                'Metric': metric,
                'Drifts': n_drifts,
                'Features': n_features,
                'Scenario': f'{n_drifts}d {n_features}f',
                'Detector': det,
                'Value': val,
            })

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / 'synthetic_heatmap_values_D1_D2_R_with_Feature_KS.csv', index=False)


def matrix_for(drift_type, metric):
    values = metric_data[(drift_type, metric)]
    mat = []
    for d in drifts:
        for f in features:
            row = values[(d, f)]
            mat.append([row[idx] for idx in detector_col_idx])
    return np.array(mat, dtype=float)


def text_color(value, vmin, vmax, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    if vmax == vmin:
        norm_value = 0.5
    else:
        norm_value = (value - vmin) / (vmax - vmin)
    r, g, b, _ = cmap(norm_value)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.50 else 'black'


def annotate(ax, mat, vmin, vmax, fontsize=CELL_VALUE_FONTSIZE):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i, f'{mat[i, j]:.2f}',
                ha='center', va='center',
                fontsize=fontsize,
                color=text_color(mat[i, j], vmin, vmax),
            )

plt.rcParams.update({
    'font.size': BASE_FONTSIZE,
    'axes.labelsize': AXIS_LABEL_FONTSIZE,
    'axes.titlesize': TITLE_FONTSIZE,
    'xtick.labelsize': XTICK_FONTSIZE,
    'ytick.labelsize': YTICK_FONTSIZE,
})

# One color scale per metric, shared between abrupt and gradual rows.
metric_ranges = {}
for metric in ['D1', 'D2', 'R']:
    combined = np.concatenate([
        matrix_for('abrupt', metric).ravel(),
        matrix_for('gradual', metric).ravel(),
    ])
    metric_ranges[metric] = (float(np.nanmin(combined)), float(np.nanmax(combined)))


def format_axis(ax, row_idx, col_idx, drift_type, show_y=True, show_x=True):
    ax.set_xticks(np.arange(len(detectors)))
    if show_x:
        ax.set_xticklabels(detectors, rotation=28, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Detector')
    else:
        ax.set_xticklabels([])

    ax.set_yticks(np.arange(len(row_labels)))
    if show_y:
        ax.set_yticklabels(row_labels)
        ax.set_ylabel('Abrupt drift' if drift_type == 'abrupt' else 'Gradual drift', labelpad=14)
    else:
        ax.set_yticklabels([])

    # subtle grid between cells
    ax.set_xticks(np.arange(-0.5, len(detectors), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which='minor', linewidth=0.15, alpha=0.18)
    ax.tick_params(which='minor', bottom=False, left=False)


# Combined 2x3 figure.
fig = plt.figure(figsize=(28, 14))
gs = GridSpec(
    2, 6,
    width_ratios=[1.0, 0.035, 1.0, 0.035, 1.0, 0.035],
    height_ratios=[1, 1],
    wspace=0.18,
    hspace=0.20,
    figure=fig,
)

images = {}
for row_idx, drift_type in enumerate(['abrupt', 'gradual']):
    for col_idx, metric in enumerate(['D1', 'D2', 'R']):
        ax = fig.add_subplot(gs[row_idx, col_idx * 2])
        mat = matrix_for(drift_type, metric)
        vmin, vmax = metric_ranges[metric]
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        images[metric] = im

        if row_idx == 0:
            ax.set_title(metric_labels[metric], pad=8)

        format_axis(
            ax,
            row_idx=row_idx,
            col_idx=col_idx,
            drift_type=drift_type,
            show_y=(col_idx == 0),
            show_x=(row_idx == 1),
        )
        annotate(ax, mat, vmin, vmax)

# One colorbar per metric, spanning both rows.
for col_idx, metric in enumerate(['D1', 'D2', 'R']):
    cax = fig.add_subplot(gs[:, col_idx * 2 + 1])
    cbar = fig.colorbar(images[metric], cax=cax)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

combined_base = OUT_DIR / 'synthetic_heatmaps_D1_D2_R_abrupt_gradual_2x3_with_Feature_KS'
fig.savefig(combined_base.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.12)
fig.savefig(combined_base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.12)
fig.savefig(combined_base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.12)
plt.close(fig)


def save_single_metric(metric):
    vmin, vmax = metric_ranges[metric]
    fig = plt.figure(figsize=(8, 14.5))
    gs = GridSpec(
        2, 2,
        width_ratios=[1.0, 0.04],
        height_ratios=[1, 1],
        wspace=0.08,
        hspace=0.26,
        figure=fig,
    )

    im = None
    for row_idx, drift_type in enumerate(['abrupt', 'gradual']):
        ax = fig.add_subplot(gs[row_idx, 0])
        mat = matrix_for(drift_type, metric)
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(metric_labels[metric], pad=8)
        format_axis(
            ax,
            row_idx=row_idx,
            col_idx=0,
            drift_type=drift_type,
            show_y=True,
            show_x=(row_idx == 1),
        )
        annotate(ax, mat, vmin, vmax, fontsize=CELL_VALUE_FONTSIZE)

    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    base = OUT_DIR / f'synthetic_heatmap_{metric}_abrupt_gradual_with_Feature_KS'
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.12)
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.12)
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


for metric in ['D1', 'D2', 'R']:
    save_single_metric(metric)

zip_path = OUT_DIR / 'synthetic_heatmaps_with_Feature_KS_package.zip'
current_script = Path(__file__).resolve()

output_names = [
    'synthetic_heatmaps_D1_D2_R_abrupt_gradual_2x3_with_Feature_KS.png',
    'synthetic_heatmaps_D1_D2_R_abrupt_gradual_2x3_with_Feature_KS.pdf',
    'synthetic_heatmaps_D1_D2_R_abrupt_gradual_2x3_with_Feature_KS.svg',
    'synthetic_heatmap_D1_abrupt_gradual_with_Feature_KS.pdf',
    'synthetic_heatmap_D1_abrupt_gradual_with_Feature_KS.png',
    'synthetic_heatmap_D1_abrupt_gradual_with_Feature_KS.svg',
    'synthetic_heatmap_D2_abrupt_gradual_with_Feature_KS.pdf',
    'synthetic_heatmap_D2_abrupt_gradual_with_Feature_KS.png',
    'synthetic_heatmap_D2_abrupt_gradual_with_Feature_KS.svg',
    'synthetic_heatmap_R_abrupt_gradual_with_Feature_KS.pdf',
    'synthetic_heatmap_R_abrupt_gradual_with_Feature_KS.png',
    'synthetic_heatmap_R_abrupt_gradual_with_Feature_KS.svg',
    'synthetic_heatmap_values_D1_D2_R_with_Feature_KS.csv',
    current_script.name,
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in output_names:
        path = OUT_DIR / name
        if path.exists():
            zf.write(path, arcname=path.name)

print('Created files:')
for name in output_names:
    path = OUT_DIR / name
    if path.exists():
        print(path)
print(zip_path)
