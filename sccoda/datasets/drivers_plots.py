import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

def load_driver_metrics(pattern):
    files = glob.glob(pattern)

    metrics = {}

    for f in files:
        key_match = re.search(r"_P(\d+)", f)
        key = int(key_match.group(1))
        metrics[key] = pd.read_csv(f)

    return dict(sorted(metrics.items(), key=lambda x: x[0]))


def extract_counts(df):
    return (
        int(df["TN"].iloc[0]),
        int(df["FP"].iloc[0]),
        int(df["FN"].iloc[0]),
        int(df["TP"].iloc[0]),
    )


def plot_confusion_matrix(ax, TN, FP, FN, TP, title=None):
    mat = np.array([[1, 0],
                    [0, 1]])

    orange = "#f4b183"
    blue   = "#c9daf8"
    cmap = ListedColormap([blue, orange])

    cm = np.array([[TP, FP],
                   [FN, TN]])

    labels = [["TP", "FP"],
              ["FN", "TN"]]

    ax.imshow(mat, cmap=cmap, aspect="equal")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold")

    ax.set_xticks([0,1]); ax.set_xticklabels(["Positive", "Negative"], fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Positive", "Negative"], fontsize=9)
    ax.set_xlabel("True", fontsize=10)
    ax.set_ylabel("Inferred", fontsize=10)

    if title:
        ax.set_title(title, fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(False)


def barplot_confusion_counts(ax, metrics_dict, xlabel="Key"):
    keys = list(metrics_dict.keys())

    TP = [int(metrics_dict[k]["TP"].iloc[0]) for k in keys]
    FP = [int(metrics_dict[k]["FP"].iloc[0]) for k in keys]
    FN = [int(metrics_dict[k]["FN"].iloc[0]) for k in keys]
    TN = [int(metrics_dict[k]["TN"].iloc[0]) for k in keys]

    bar_width = 0.2
    x = np.arange(len(keys))

    ax.bar(x - 1.5*bar_width, TP, width=bar_width, label="TP", color="#9a83f4")
    ax.bar(x - 0.5*bar_width, FP, width=bar_width, label="FP", color="#f3c9f8")
    ax.bar(x + 0.5*bar_width, FN, width=bar_width, label="FN", color="#eef8c9")
    ax.bar(x + 1.5*bar_width, TN, width=bar_width, label="TN", color="#f4b183")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=8)


def plot_roc(ax, metrics_dict, type_label="K"):
    keys = list(metrics_dict.keys())
    TPR, FPR = [], []

    for k in keys:
        df = metrics_dict[k]
        TP = int(df["TP"].iloc[0])
        FP = int(df["FP"].iloc[0])
        FN = int(df["FN"].iloc[0])
        TN = int(df["TN"].iloc[0])

        tpr = TP / (TP + FN) if (TP + FN) else 0
        fpr = FP / (FP + TN) if (FP + TN) else 0

        TPR.append(tpr)
        FPR.append(fpr)

    ax.plot(FPR, TPR, marker="o")
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)

    for fpr, tpr, k in zip(FPR, TPR, keys):
        ax.text(fpr, tpr, f"{type_label}={k}", fontsize=8)

    ax.set_xlabel("FPR", fontsize=10)
    ax.set_ylabel("TPR", fontsize=10)


def make_driver_figure_onepattern(pattern, outfile_prefix="figure", key_label="N"):
    metrics = load_driver_metrics(pattern)
    keys = list(metrics.keys())

    first_four = keys[:4]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.4], wspace=0.25, hspace=0.35)

    panels = ["(a)", "(b)", "(c)", "(d)"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for label, key, pos in zip(panels, first_four, positions):
        ax = fig.add_subplot(gs[pos])
        TN, FP, FN, TP = extract_counts(metrics[key])
        plot_confusion_matrix(ax, TN, FP, FN, TP, title=f"{label} {key_label}={key}")


    ax_e = fig.add_subplot(gs[0, 2])
    barplot_confusion_counts(ax_e, metrics, xlabel=key_label)
    ax_e.set_title("(e) Confusion Counts", fontsize=12)

    ax_f = fig.add_subplot(gs[1, 2])
    plot_roc(ax_f, metrics, type_label=key_label)
    ax_f.set_title("(f) TPR–FPR", fontsize=12)

    # Save
    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern"
    )
    parser.add_argument(
        "--out",
        default="figure"
    )
    parser.add_argument(
    "--label"
)
    args = parser.parse_args()

    make_driver_figure_onepattern(args.pattern, outfile_prefix=args.out, key_label=args.label)
if __name__ == "__main__":
    main()