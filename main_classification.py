"""
Task 2 — Multimodal Classification: Abnormal vs Normal CXR
Entry point: python main_classification.py
"""
import os
import sys
import json
import warnings
import numpy as np

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR
from classification.label_utils import load_labels, get_case_paths

OUTPUT_DIR = "outputs/classification"

N_FOLDS  = 5
EPOCHS   = 120    # more epochs with smaller model for better convergence
N_BINS   = 20     # fewer temporal bins — reduces sequence length for small N
D_MODEL  = 32     # reduced hidden dim — prevents overfitting on N=50


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Labels ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading labels")
    print("=" * 60)
    case_paths = get_case_paths(DATASET_DIR)
    y = load_labels(DATASET_DIR)
    n_cases = len(y)
    print(f"Total cases: {n_cases}")

    if len(np.unique(y)) < 2:
        print("[ERROR] Only one class found — cannot train a classifier.")
        return

    # ── Step 2: Feature extraction ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Building features")
    print("=" * 60)

    print("\n--- Image features ---")
    from classification.image_encoder import build_image_features
    X_img, img_feat_names = build_image_features(case_paths)

    print("\n--- Gaze features ---")
    from classification.gaze_encoder import build_all_gaze_sequences, build_gaze_static_features
    gaze_sequences = build_all_gaze_sequences(case_paths, n_bins=N_BINS)
    X_gaze_static, gaze_feat_names = build_gaze_static_features(case_paths)

    print("\n--- Speech features ---")
    from classification.speech_encoder import build_all_speech_sequences, build_speech_static_features
    speech_sequences = build_all_speech_sequences(case_paths, n_bins=N_BINS)
    X_speech_static, speech_feat_names = build_speech_static_features(case_paths)

    print("\n--- Alignment features ---")
    alignment_features = None
    try:
        import cv2
        from preprocessing.gaze_processing import load_gaze, define_aois
        from preprocessing.speech_processing import SpeechEncoder
        from preprocessing.cross_modal import compute_alignment_features
        from config import GAZE_FILE, TRANSCRIPTION_FILE, IMAGE_FILE

        from preprocessing.gaze_processing import extract_gaze_features, map_aoi
        enc = SpeechEncoder()
        align_rows = []
        for cp in case_paths:
            gaze_df  = load_gaze(os.path.join(cp, GAZE_FILE))
            trans_df = enc.load_transcription(os.path.join(cp, TRANSCRIPTION_FILE))
            img      = cv2.imread(os.path.join(cp, IMAGE_FILE))
            h, w = img.shape[:2] if img is not None else (512, 512)
            aois = define_aois(w, h)
            aoi_sequence = [map_aoi(x, y, aois) for x, y in
                            zip(gaze_df["x"].values, gaze_df["y"].values)]
            af   = compute_alignment_features(gaze_df, trans_df, aoi_sequence, aois)
            align_rows.append([
                af.get("gaze_to_speech_lag", 0),
                af.get("revisits_before_mention", 0),
                af.get("mentioned_aoi_dwell_fraction", 0),
                af.get("unmentioned_aoi_dwell_fraction", 0),
            ])
        alignment_features = np.array(align_rows, dtype=np.float32)
        alignment_features = np.nan_to_num(alignment_features, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Alignment features: {alignment_features.shape}")
    except Exception as e:
        print(f"  [warn] Alignment features skipped: {e}")

    print(f"\nFeature shapes summary:")
    print(f"  Image       : {X_img.shape}")
    print(f"  Gaze seqs   : {gaze_sequences.shape}")
    print(f"  Gaze static : {X_gaze_static.shape}")
    print(f"  Speech seqs : {speech_sequences.shape}")
    print(f"  Speech static: {X_speech_static.shape}")

    # ── Step 3: Simple baselines ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Simple baselines (sklearn)")
    print("=" * 60)
    from classification.simple_baselines import run_simple_baselines
    baseline_results = run_simple_baselines(
        X_img, X_gaze_static, X_speech_static, y,
        alignment_features=alignment_features,
        n_folds=N_FOLDS,
    )

    # ── Step 4: Cross-attention fusion model ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Cross-attention fusion model (PyTorch)")
    print("=" * 60)

    gaze_feat_dim   = gaze_sequences.shape[2]   # 9
    speech_feat_dim = speech_sequences.shape[2]  # 16
    image_feat_dim  = X_img.shape[1]

    base_model_kwargs = {
        "image_feat_dim":  image_feat_dim,
        "gaze_feat_dim":   gaze_feat_dim,
        "speech_feat_dim": speech_feat_dim,
        "d_model":         D_MODEL,
    }

    all_data = {
        "img":    X_img,
        "gaze":   gaze_sequences,
        "speech": speech_sequences,
    }

    from classification.trainer import run_modality_ablation
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fusion_results = run_modality_ablation(
            all_data, y,
            base_model_kwargs=base_model_kwargs,
            n_folds=N_FOLDS,
            epochs=EPOCHS,
        )

    # ── Step 5: Comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Fusion vs Baselines comparison")
    print("=" * 60)
    print(f"{'Condition':<30} {'Baseline AUC':>13}  {'Fusion AUC':>11}")
    print("-" * 60)
    bl_by_label = {r["label"]: r for r in baseline_results}
    fu_by_label = {r["label"]: r for r in fusion_results}
    for lbl in [r["label"] for r in baseline_results]:
        bl  = bl_by_label.get(lbl, {}).get("auc", float("nan"))
        fu  = fu_by_label.get(lbl, {}).get("auc", float("nan"))
        print(f"{lbl:<30} {bl:>13.4f}  {fu:>11.4f}")

    # ── Step 6: Significance testing ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Statistical significance")
    print("=" * 60)
    from classification.significance import paired_significance_tests
    sig_results = paired_significance_tests(fusion_results)

    # Modality dropout (uses the full model's AUC)
    full_res = fu_by_label.get("Image + Gaze + Speech", {})
    full_auc = full_res.get("auc", 0.5)

    try:
        from classification.significance import modality_dropout_test
        from classification.fusion_model import MultimodalClassifier
        dropout_results = modality_dropout_test(
            all_data, y,
            model_class=MultimodalClassifier,
            base_model_kwargs=base_model_kwargs,
            full_auc=full_auc,
            n_folds=N_FOLDS,
            epochs=EPOCHS,
        )
    except Exception as e:
        print(f"  [warn] Dropout test skipped: {e}")
        dropout_results = []

    # ── Step 7: Attention analysis ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7: Cross-attention analysis")
    print("=" * 60)
    from classification.attention_analysis import analyze_cross_attention, error_analysis

    # Collect all fold attention weights from the full model condition
    full_attn_list = []
    for r in fusion_results:
        if r["label"] == "Image + Gaze + Speech":
            for fold_attn in r.get("attention_weights", []):
                full_attn_list.append(fold_attn)

    attn_analysis = analyze_cross_attention(
        full_attn_list, y,
        aoi_names=["left_lung", "right_lung", "heart",
                   "lower_left", "lower_right", "background"],
    ) if full_attn_list else {}

    # Error analysis on full model predictions
    if full_res:
        error_indices, error_profiles = error_analysis(
            full_res.get("all_val_probs", np.zeros(n_cases)),
            full_res.get("all_val_labels", y),
            case_paths,
            features={"gaze_static": X_gaze_static, "speech_static": X_speech_static},
        )
    else:
        error_indices, error_profiles = np.array([]), []

    # ── Step 8: Visualisations ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 8: Generating visualisations")
    print("=" * 60)
    from classification.viz.classification_plots import (
        plot_ablation_bars, plot_roc_curves, plot_confusion_matrices,
        plot_cross_attention_heatmaps, plot_modality_dropout,
        plot_training_curves, plot_modality_radar,
    )

    plot_ablation_bars(fusion_results)
    plot_roc_curves(fusion_results)
    plot_confusion_matrices(fusion_results)

    if attn_analysis:
        plot_cross_attention_heatmaps(attn_analysis)

    if dropout_results:
        plot_modality_dropout(dropout_results, full_auc)

    # Training curves for full model
    full_hist = full_res.get("histories", []) if full_res else []
    if full_hist:
        plot_training_curves(full_hist, condition_label="Image + Gaze + Speech")

    plot_modality_radar(fusion_results)

    # Feature importance from best baseline (RF)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        all_static_names = img_feat_names + gaze_feat_names + speech_feat_names
        X_all_static = np.concatenate([X_img, X_gaze_static, X_speech_static], axis=1)
        X_all_static = np.nan_to_num(X_all_static)
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_all_static)
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_sc, y)
        from classification.viz.classification_plots import plot_feature_importance
        plot_feature_importance(rf.feature_importances_, all_static_names)
    except Exception as e:
        print(f"  [warn] Feature importance plot skipped: {e}")

    # ── Step 9: Save summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 9: Writing summary")
    print("=" * 60)
    _write_summary(
        output_dir=OUTPUT_DIR,
        y=y,
        baseline_results=baseline_results,
        fusion_results=fusion_results,
        sig_results=sig_results,
        dropout_results=dropout_results,
        error_profiles=error_profiles,
        full_auc=full_auc,
    )
    print("\nDone.  All outputs in", OUTPUT_DIR)


# ── Summary writer ────────────────────────────────────────────────────────────

def _write_summary(output_dir, y, baseline_results, fusion_results,
                   sig_results, dropout_results, error_profiles, full_auc):
    lines = []
    lines.append("=" * 65)
    lines.append("Task 2 — Multimodal CXR Classification: Abnormal vs Normal")
    lines.append("=" * 65)

    n_normal   = int((y == 0).sum())
    n_abnormal = int((y == 1).sum())
    lines.append(f"\nDataset: {len(y)} cases  |  Normal: {n_normal}  Abnormal: {n_abnormal}")

    lines.append("\n--- Simple Baselines (sklearn) ---")
    lines.append(f"{'Condition':<30} {'Model':<16} {'AUC':>7}  {'F1':>6}")
    for r in baseline_results:
        lines.append(f"{r['label']:<30} {r['clf_name']:<16} "
                     f"{r['auc']:>7.4f}  {r['f1']:>6.4f}")

    lines.append("\n--- Cross-Attention Fusion Model ---")
    lines.append(f"{'Condition':<30} {'AUC':>7}  {'±':>5}  {'F1':>6}  {'Bal.Acc':>8}  {'Sens':>6}  {'Spec':>6}")
    for r in fusion_results:
        lines.append(f"{r['label']:<30} {r['auc']:>7.4f}  "
                     f"±{r['std_auc']:>5.3f}  {r['f1']:>6.4f}  "
                     f"{r['balanced_acc']:>8.4f}  "
                     f"{r['sensitivity']:>6.4f}  {r['specificity']:>6.4f}")

    lines.append("\n--- Significance Tests ---")
    for s in sig_results:
        sig_str = "*" if s["significant"] else "ns"
        lines.append(f"  {s['description']:<45} "
                     f"ΔAUC={s['delta_auc']:+.4f}  p={s['p_value']:.4f}  {sig_str}")

    if dropout_results:
        lines.append("\n--- Modality Dropout ---")
        for d in dropout_results:
            lines.append(f"  Drop {d['dropped']:<10} AUC={d['auc_without']:.4f}  "
                         f"ΔAUC={d['delta']:+.4f}  dep={d['dependency']:.4f}")

    if error_profiles:
        lines.append(f"\n--- Error Analysis ({len(error_profiles)} errors) ---")
        for ep in error_profiles:
            lines.append(f"  {ep['case'].split('/')[-1]}  {ep['error_type']}  "
                         f"prob={ep['prob']:.3f}")

    best_fusion = max(fusion_results, key=lambda r: r["auc"])
    lines.append(f"\n--- Key Finding ---")
    lines.append(f"Best condition: {best_fusion['label']}  "
                 f"AUC={best_fusion['auc']:.4f}")

    summary_text = "\n".join(lines)
    path = os.path.join(output_dir, "classification_summary.txt")
    with open(path, "w") as f:
        f.write(summary_text)
    print(f"Saved: {path}")

    # Also save machine-readable JSON
    import json as _json
    json_safe = []
    for r in fusion_results:
        row = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in r.items()
               if k not in ("fpr", "tpr", "all_val_probs", "all_val_labels",
                            "attention_weights", "histories")}
        json_safe.append(row)
    json_path = os.path.join(output_dir, "fusion_results.json")
    with open(json_path, "w") as f:
        _json.dump(json_safe, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
