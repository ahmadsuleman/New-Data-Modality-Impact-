"""
classification/viz/classification_plots.py
All visualisations for the classification pipeline.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

OUTPUT_DIR = "outputs/classification"


def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── 1. Ablation bar chart ────────────────────────────────────────────────────

def plot_ablation_bars(ablation_results: list):
    labels   = [r["label"] for r in ablation_results]
    aucs     = [r["auc"]   for r in ablation_results]
    stds     = [r["std_auc"] for r in ablation_results]
    f1s      = [r["f1"]    for r in ablation_results]
    bal_accs = [r["balanced_acc"] for r in ablation_results]

    x      = np.arange(len(labels))
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, aucs,     width, label="AUC-ROC", color=colors[0],
           yerr=stds, capsize=4, alpha=0.85)
    ax.bar(x,         f1s,      width, label="F1",       color=colors[1], alpha=0.85)
    ax.bar(x + width, bal_accs, width, label="Bal. Acc", color=colors[2], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Modality Ablation — Classification Performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
    return _save(fig, "ablation_comparison.png")


# ── 2. ROC curves ────────────────────────────────────────────────────────────

def plot_roc_curves(ablation_results: list):
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = plt.cm.tab10.colors

    for i, r in enumerate(ablation_results):
        fpr = r.get("fpr", [0, 1])
        tpr = r.get("tpr", [0, 1])
        auc = r.get("auc", 0.5)
        ax.plot(fpr, tpr, color=palette[i % 10],
                label=f"{r['label']} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Modality Ablation")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "roc_curves.png")


# ── 3. Confusion matrices ────────────────────────────────────────────────────

def plot_confusion_matrices(ablation_results: list):
    n  = len(ablation_results)
    nc = 3
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.5 * nr))
    axes = np.array(axes).ravel()

    for i, r in enumerate(ablation_results):
        y_true = r.get("all_val_labels")
        y_prob = r.get("all_val_probs")
        if y_true is None or y_prob is None:
            continue
        y_pred = (y_prob >= 0.5).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax     = axes[i]

        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for row in range(2):
            for col in range(2):
                total = cm.sum()
                pct   = 100 * cm[row, col] / max(total, 1)
                ax.text(col, row, f"{cm[row, col]}\n({pct:.0f}%)",
                        ha="center", va="center",
                        color="white" if cm[row, col] > cm.max() / 2 else "black",
                        fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticklabels(["Normal", "Abnormal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(r["label"], fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion Matrices — Modality Ablation", y=1.01)
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


# ── 4. Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(importance_scores: np.ndarray, feature_names: list):
    """Horizontal bar chart, colored by modality group."""
    modality_colors = {}
    for name in feature_names:
        if name.startswith("img"):
            modality_colors[name] = "#4C72B0"
        elif name.startswith("gaze"):
            modality_colors[name] = "#DD8452"
        elif name.startswith("speech"):
            modality_colors[name] = "#55A868"
        else:
            modality_colors[name] = "#8172B2"

    order  = np.argsort(importance_scores)[::-1][:30]  # top 30
    scores = importance_scores[order]
    names  = [feature_names[i] for i in order]
    colors = [modality_colors.get(n, "#999") for n in names]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), scores[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (by modality)")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Image"),
        Patch(facecolor="#DD8452", label="Gaze"),
        Patch(facecolor="#55A868", label="Speech"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "feature_importance.png")


# ── 5. Cross-attention heatmaps ──────────────────────────────────────────────

def plot_cross_attention_heatmaps(attention_analysis: dict):
    keys = list(attention_analysis.keys())
    if not keys:
        print("No attention data to plot.")
        return

    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 2, figsize=(12, 4 * n_keys))
    if n_keys == 1:
        axes = [axes]

    for row_i, key in enumerate(keys):
        info   = attention_analysis[key]
        norm_m = np.array(info["normal_mean"]).reshape(1, -1)
        abn_m  = np.array(info["abnormal_mean"]).reshape(1, -1)

        for col_j, (mat, title) in enumerate([
            (norm_m,  f"{info['label']}\nNormal cases"),
            (abn_m,   f"{info['label']}\nAbnormal cases"),
        ]):
            ax  = axes[row_i][col_j]
            im  = ax.imshow(mat, aspect="auto", cmap="hot", vmin=0)
            plt.colorbar(im, ax=ax, fraction=0.03)
            ax.set_yticks([])
            ax.set_xlabel("Time bin")
            ax.set_title(title, fontsize=9)

    fig.suptitle("Cross-Attention Heatmaps (mean over cases)", y=1.01)
    fig.tight_layout()
    return _save(fig, "attention_heatmaps.png")


# ── 6. Modality dropout bar ──────────────────────────────────────────────────

def plot_modality_dropout(dropout_results: list, full_auc: float):
    labels  = [r["dropped"].capitalize() for r in dropout_results]
    deltas  = [r["delta"] for r in dropout_results]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, deltas, color=colors[:len(labels)], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("AUC drop when modality removed")
    ax.set_title(f"Modality Dependency (full AUC = {full_auc:.3f})")
    ax.grid(axis="y", alpha=0.3)
    for i, (lbl, d) in enumerate(zip(labels, deltas)):
        ax.text(i, d + 0.005, f"{d:+.3f}", ha="center", va="bottom", fontsize=9)
    return _save(fig, "modality_dropout.png")


# ── 7. Training curves ───────────────────────────────────────────────────────

def plot_training_curves(training_histories: list, condition_label: str = "Full model"):
    """Plot loss and AUC per fold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for fold_i, hist in enumerate(training_histories):
        axes[0].plot(hist.get("train_loss", []), alpha=0.6, label=f"Fold {fold_i + 1}")
        axes[1].plot(hist.get("val_auc", []),   alpha=0.6, label=f"Fold {fold_i + 1}")

    axes[0].set_title(f"Training Loss — {condition_label}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"Validation AUC — {condition_label}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "training_curves.png")


# ── 8. Radar chart ───────────────────────────────────────────────────────────

def plot_modality_radar(ablation_results: list):
    from matplotlib.patches import FancyArrowPatch
    metrics_keys   = ["auc", "f1", "balanced_acc", "sensitivity", "specificity"]
    metrics_labels = ["AUC", "F1", "Bal. Acc", "Sensitivity", "Specificity"]
    n_metrics = len(metrics_keys)
    angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles   += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={"polar": True})
    palette = plt.cm.tab10.colors

    for i, r in enumerate(ablation_results):
        values = [r.get(k, 0) for k in metrics_keys]
        values += values[:1]
        ax.plot(angles, values, color=palette[i % 10],
                linewidth=2, label=r["label"])
        ax.fill(angles, values, color=palette[i % 10], alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Modality Ablation Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    return _save(fig, "modality_radar.png")
