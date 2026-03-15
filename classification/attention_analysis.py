"""
classification/attention_analysis.py
Interpretability — cross-attention weight analysis and error analysis.
"""
import numpy as np


AOI_NAMES = ["left_lung", "right_lung", "heart", "lower_left", "lower_right", "background"]


def analyze_cross_attention(attention_weights_list: list,
                            labels: np.ndarray,
                            aoi_names: list = AOI_NAMES):
    """
    Analyse attention patterns for normal vs abnormal cases.

    Parameters
    ----------
    attention_weights_list : list of per-case dicts with keys
                             img_gaze, img_speech, gaze_speech
                             Each value is np.ndarray (heads, 1, n_bins) or similar.
    labels                 : np.ndarray, shape (n,)
    aoi_names              : list[str]

    Returns
    -------
    summary : dict
    """
    if not attention_weights_list:
        return {}

    normal_idx   = np.where(labels == 0)[0]
    abnormal_idx = np.where(labels == 1)[0]

    def _get_attn_seq(attn_list, key):
        """Extract mean attention over heads → (n_cases, n_bins)."""
        rows = []
        for attn in attn_list:
            w = attn.get(key)
            if w is None:
                continue
            # w shape: (batch, heads, q_len, k_len) or (heads, q_len, k_len)
            w = np.array(w)
            if w.ndim == 4:
                w = w.mean(axis=(0, 1, 2))   # collapse batch/heads/q → (k_len,)
            elif w.ndim == 3:
                w = w.mean(axis=(0, 1))
            elif w.ndim == 2:
                w = w.mean(axis=0)
            rows.append(w)
        return np.stack(rows) if rows else None

    summary = {}

    for key, label in [("img_gaze",    "image→gaze attention"),
                        ("img_speech",  "image→speech attention"),
                        ("gaze_speech", "gaze→speech attention")]:
        seq = _get_attn_seq(attention_weights_list, key)
        if seq is None or len(seq) == 0:
            continue

        n_bins = seq.shape[1] if seq.ndim > 1 else len(seq)

        def _safe_mean(idx):
            valid = [i for i in idx if i < len(seq)]
            return seq[valid].mean(axis=0) if valid else np.zeros(n_bins)

        normal_mean   = _safe_mean(normal_idx)
        abnormal_mean = _safe_mean(abnormal_idx)

        summary[key] = {
            "label":         label,
            "normal_mean":   normal_mean,
            "abnormal_mean": abnormal_mean,
            "delta":         abnormal_mean - normal_mean,
            "peak_bin_normal":   int(np.argmax(normal_mean)),
            "peak_bin_abnormal": int(np.argmax(abnormal_mean)),
        }

        print(f"\n[{label}]")
        print(f"  Peak bin — Normal: {summary[key]['peak_bin_normal']}, "
              f"Abnormal: {summary[key]['peak_bin_abnormal']}")
        print(f"  Mean attention — Normal: {normal_mean.mean():.4f}, "
              f"Abnormal: {abnormal_mean.mean():.4f}")

    return summary


def error_analysis(predictions: np.ndarray,
                   labels: np.ndarray,
                   case_paths: list,
                   features: dict):
    """
    Identify and characterise misclassified cases.

    Parameters
    ----------
    predictions : np.ndarray of probabilities (n,)
    labels      : np.ndarray of true labels (n,)
    case_paths  : list[str]
    features    : dict with img, gaze_static, speech_static arrays

    Returns
    -------
    error_indices : np.ndarray
    error_profiles: list of dicts
    """
    pred_labels = (predictions >= 0.5).astype(int)
    errors = np.where(pred_labels != labels)[0]
    fn_idx = np.where((pred_labels == 0) & (labels == 1))[0]  # missed abnormals
    fp_idx = np.where((pred_labels == 1) & (labels == 0))[0]  # false alarms

    print(f"\nError Analysis: {len(errors)} misclassified cases "
          f"({len(fn_idx)} FN, {len(fp_idx)} FP)")

    error_profiles = []
    for idx in errors:
        error_type = "FN (missed abnormal)" if labels[idx] == 1 else "FP (false alarm)"
        profile = {
            "index":      int(idx),
            "case":       case_paths[idx] if idx < len(case_paths) else "unknown",
            "error_type": error_type,
            "prob":       float(predictions[idx]),
            "true_label": int(labels[idx]),
        }
        # Add feature values if available
        if "gaze_static" in features and idx < len(features["gaze_static"]):
            g = features["gaze_static"][idx]
            profile["gaze_entropy"]  = float(g[5]) if len(g) > 5 else 0.0
            profile["gaze_fixcount"] = float(g[0]) if len(g) > 0 else 0.0
        if "speech_static" in features and idx < len(features["speech_static"]):
            s = features["speech_static"][idx]
            n_emb = len(s) - 4  # last 4 are keyword features
            profile["speech_finding_mentions"] = float(s[n_emb + 1]) if len(s) > n_emb + 1 else 0.0
            profile["speech_negation_count"]   = float(s[n_emb + 2]) if len(s) > n_emb + 2 else 0.0

        error_profiles.append(profile)
        print(f"  [{idx:3d}] {error_type:<25}  prob={predictions[idx]:.3f}  "
              f"case={profile['case'].split('/')[-1]}")

    return errors, error_profiles
