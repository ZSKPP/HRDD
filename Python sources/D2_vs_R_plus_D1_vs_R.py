# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import numpy as np
from pathlib import Path

# ============================================================
# USTAWIENIA PDF / COREL
# ============================================================

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.compression"] = 0

# Jeżeli Corel źle pokazuje PDF, otwieraj SVG albo ustaw AREA_ALPHA = 1.0.
# AREA_ALPHA = 0.20 daje efekt podobny do pierwotnego wykresu.
AREA_ALPHA = 0.20

OUTPUT_DIR = Path(__file__).resolve().parent

# =========================
# DANE
# Kolejność scenariuszy:
# 3d|30f, 3d|60f, 3d|90f,
# 5d|30f, 5d|60f, 5d|90f,
# 10d|30f, 10d|60f, 10d|90f,
# 15d|30f, 15d|60f, 15d|90f
# =========================

DETECTORS = ["HRDD", "ADWIN", "EDDM", "DDM", "OCDD"]

D2_gradual = {
    "HRDD": [3.39, 5.60, 13.63, 2.28, 7.88, 12.00, 1.80, 6.09, 6.42, 1.05, 3.30, 4.21],
    "ADWIN": [3.82, 6.49, 4.48, 5.92, 8.06, 5.06, 7.78, 7.83, 9.68, 7.56, 11.03, 15.57],
    "EDDM": [28.72, 41.06, 31.89, 31.26, 43.34, 35.48, 34.47, 28.60, 29.00, 40.32, 26.51, 25.91],
    "DDM": [16.04, 10.53, 14.06, 22.78, 26.62, 16.22, 20.95, 23.67, 11.13, 19.38, 19.25, 38.51],
    "OCDD": [1.33, 1.18, 1.71, 0.90, 0.74, 1.32, 0.95, 1.17, 1.10, 0.90, 1.03, 0.77],
}

D2_abrupt = {
    "HRDD": [0.30, 1.28, 8.68, 0.28, 3.78, 5.94, 1.10, 2.78, 6.09, 0.44, 2.25, 3.08],
    "ADWIN": [3.18, 2.50, 3.99, 3.30, 2.64, 3.18, 5.38, 5.71, 5.21, 6.02, 8.79, 13.29],
    "EDDM": [29.63, 46.33, 32.41, 27.38, 39.00, 28.74, 36.78, 33.01, 19.66, 36.93, 27.89, 27.45],
    "DDM": [7.24, 2.33, 7.72, 10.28, 12.10, 8.70, 19.61, 13.66, 17.36, 17.94, 19.57, 30.40],
    "OCDD": [0.51, 0.39, 0.39, 0.32, 0.12, 0.04, 0.66, 0.57, 0.53, 0.59, 0.42, 0.37],
}

# D1 dodane dla drugiego wykresu R = f(D1)
D1_gradual = {
    "HRDD": [5.07, 6.43, 3.61, 2.76, 3.42, 4.51, 1.54, 2.58, 2.68, 0.89, 1.82, 0.60],
    "ADWIN": [15.35, 17.08, 16.09, 10.11, 10.51, 10.05, 4.74, 4.52, 4.78, 4.10, 3.84, 4.20],
    "EDDM": [17.67, 21.19, 16.37, 11.71, 11.34, 9.24, 6.30, 5.76, 5.82, 4.13, 3.94, 4.30],
    "DDM": [3.02, 3.00, 3.90, 2.95, 3.56, 3.45, 3.14, 2.80, 3.65, 2.61, 2.27, 2.52],
    "OCDD": [17.17, 10.92, 8.18, 9.76, 4.68, 4.76, 0.72, 3.11, 2.15, 3.31, 2.42, 1.96],
}

D1_abrupt = {
    "HRDD": [2.66, 3.05, 2.29, 1.55, 2.37, 2.23, 0.81, 1.37, 0.92, 0.36, 0.90, 0.79],
    "ADWIN": [15.09, 13.89, 16.73, 8.74, 9.69, 9.76, 4.15, 4.24, 4.13, 3.48, 3.29, 3.73],
    "EDDM": [20.53, 21.96, 19.26, 12.06, 11.79, 12.16, 6.35, 6.36, 5.83, 4.04, 4.22, 4.14],
    "DDM": [1.87, 2.43, 2.39, 1.83, 2.35, 2.09, 2.42, 2.21, 2.70, 2.04, 2.05, 2.14],
    "OCDD": [19.96, 18.57, 13.61, 11.32, 9.93, 7.70, 5.69, 4.69, 3.94, 3.69, 3.20, 2.42],
}

R_gradual = {
    "HRDD": [0.20, 0.18, 0.16, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.06, 0.05, 0.05],
    "ADWIN": [0.45, 0.44, 0.42, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35],
    "EDDM": [0.75, 0.77, 0.78, 0.74, 0.75, 0.76, 0.73, 0.72, 0.71, 0.70, 0.69, 0.68],
    "DDM": [0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77],
    "OCDD": [0.62, 0.63, 0.64, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53],
}

R_abrupt = {
    "HRDD": [0.18, 0.17, 0.16, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.06, 0.05, 0.05],
    "ADWIN": [0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33],
    "EDDM": [0.72, 0.73, 0.74, 0.71, 0.72, 0.73, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65],
    "DDM": [0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55],
    "OCDD": [0.74, 0.75, 0.76, 0.73, 0.72, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65],
}

# =========================
# KOLORY OBSZARÓW I PUNKTÓW
# Oba wykresy korzystają z tych samych kolorów.
# =========================

AREA_COLORS = {
    "HRDD": "#ff7f0e",   # orange
    "ADWIN": "#d62728",  # red
    "EDDM": "#8c564b",   # brown
    "DDM": "#7f7f7f",    # gray
    "OCDD": "#17becf",   # cyan/turquoise
}

POINT_COLORS = AREA_COLORS.copy()

# Jeżeli chcesz inne położenie napisów na wykresach, zmień tutaj.
LABEL_OFFSETS = {
    "HRDD": (0.0, 0.00),
    "ADWIN": (0.0, 0.00),
    "EDDM": (0.0, 0.00),
    "DDM": (0.0, 0.00),
    "OCDD": (0.0, 0.00),
}

# =========================
# FUNKCJE RYSUJĄCE
# =========================

def draw_hull(ax, x, y, label):
    points = np.column_stack((x, y))

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_closed = np.vstack([hull_points, hull_points[0]])

    ax.fill(
        hull_closed[:, 0],
        hull_closed[:, 1],
        facecolor=AREA_COLORS[label],
        edgecolor=AREA_COLORS[label],
        alpha=AREA_ALPHA,
        linewidth=1.2,
        zorder=1,
    )

    ax.plot(
        hull_closed[:, 0],
        hull_closed[:, 1],
        color=AREA_COLORS[label],
        linewidth=1.2,
        zorder=2,
    )

    dx, dy = LABEL_OFFSETS[label]
    ax.text(
        np.mean(x) + dx,
        np.mean(y) + dy,
        label,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        zorder=5,
    )


def add_common_style(ax, xlabel, title):
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("R", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.tick_params(axis="both", labelsize=12)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=9,
            label="Gradual drift",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=9,
            label="Abrupt drift",
        ),
    ]

    legend = ax.legend(handles=legend_elements, prop={"size": 12, "weight": "bold"})
    for text in legend.get_texts():
        text.set_fontweight("bold")


def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def plot_tradeoff(x_gradual, x_abrupt, xlabel, title, output_stem):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Najpierw obszary, potem punkty.
    for method in DETECTORS:
        x = x_gradual[method] + x_abrupt[method]
        y = R_gradual[method] + R_abrupt[method]
        draw_hull(ax, x, y, method)

    for method in DETECTORS:
        ax.scatter(
            x_gradual[method],
            R_gradual[method],
            marker="o",
            s=40,
            color=POINT_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        ax.scatter(
            x_abrupt[method],
            R_abrupt[method],
            marker="s",
            s=40,
            color=POINT_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )

    add_common_style(ax, xlabel, title)
    fig.tight_layout()
    save_figure(fig, output_stem)
    return fig, ax


# =========================
# PLOT 1: R = f(D2)
# =========================

plot_tradeoff(
    D2_gradual,
    D2_abrupt,
    xlabel="D2",
    title="Trade-off between drift coverage (D2) and alarm consistency (R)",
    output_stem="d2_vs_r_scatter_hull",
)

# =========================
# PLOT 2: R = f(D1)  <-- NOWY WYKRES
# =========================

plot_tradeoff(
    D1_gradual,
    D1_abrupt,
    xlabel="D1",
    title="Trade-off between alarm localization (D1) and alarm consistency (R)",
    output_stem="d1_vs_r_scatter_hull",
)

plt.show()
