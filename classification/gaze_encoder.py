"""
classification/gaze_encoder.py
Gaze branch — temporal sequences and static features.
"""
import os
import numpy as np


def build_gaze_sequence(gaze_df, aois: dict, n_bins: int = 30) -> np.ndarray:
    """
    Build temporal gaze sequence of shape (n_bins, feature_dim).

    Per-bin features
    ----------------
    - AOI one-hot (6 dims: left_lung, right_lung, heart, lower_left, lower_right, background)
    - mean fixation duration in bin (1)
    - mean saccade velocity in bin (1)
    - mean pupil size in bin (1)
    Total: 9 dims
    """
    from preprocessing.gaze_processing import map_aoi

    aoi_names = ["left_lung", "right_lung", "heart", "lower_left", "lower_right", "background"]
    n_aois = len(aoi_names)
    feat_dim = n_aois + 3  # 9

    if gaze_df is None or len(gaze_df) == 0:
        return np.zeros((n_bins, feat_dim), dtype=np.float32)

    t = gaze_df["timestamp_sec"].values
    x = gaze_df["x"].values
    y = gaze_df["y"].values
    pupil = gaze_df["pupil_mm"].values if "pupil_mm" in gaze_df.columns else np.zeros(len(t))

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_max = t_min + 1e-6
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)

    # Velocities (px/sec) between consecutive samples
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-6, dt)
    dx = np.diff(x.astype(float))
    dy = np.diff(y.astype(float))
    velocities = np.sqrt(dx**2 + dy**2) / dt  # shape (n-1,)

    result = np.zeros((n_bins, feat_dim), dtype=np.float32)
    for b in range(n_bins):
        mask = (t >= bin_edges[b]) & (t < bin_edges[b + 1])
        if not mask.any():
            continue
        xi, yi = x[mask], y[mask]
        pi = pupil[mask]

        # AOI one-hot — use midpoint of bin
        mid_x = float(xi.mean())
        mid_y = float(yi.mean())
        aoi_label = map_aoi(mid_x, mid_y, aois)
        if aoi_label in aoi_names:
            result[b, aoi_names.index(aoi_label)] = 1.0
        else:
            result[b, aoi_names.index("background")] = 1.0

        # Mean fixation duration: approximate as mean inter-sample interval * 1000
        idx = np.where(mask)[0]
        if len(idx) > 1:
            dts_bin = np.diff(t[idx])
            result[b, n_aois] = float(dts_bin.mean() * 1000)  # ms
        else:
            result[b, n_aois] = 0.0

        # Mean saccade velocity in bin
        vel_mask = mask[:-1] & mask[1:]  # velocities indexed at earlier point
        if vel_mask.any():
            result[b, n_aois + 1] = float(velocities[vel_mask[:-1] if len(vel_mask) > len(velocities) else vel_mask[:len(velocities)]].mean())

        # Mean pupil size
        result[b, n_aois + 2] = float(pi.mean())

    return result


def build_all_gaze_sequences(case_paths: list, n_bins: int = 30):
    """
    Returns tensor (n_cases, n_bins, gaze_feat_dim).
    """
    import cv2
    from preprocessing.gaze_processing import load_gaze, define_aois
    from config import GAZE_FILE, IMAGE_FILE

    sequences = []
    for cp in case_paths:
        gaze_df = load_gaze(os.path.join(cp, GAZE_FILE))
        img = cv2.imread(os.path.join(cp, IMAGE_FILE))
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 512, 512
        aois = define_aois(w, h)
        seq = build_gaze_sequence(gaze_df, aois, n_bins=n_bins)
        sequences.append(seq)

    result = np.stack(sequences).astype(np.float32)
    print(f"Gaze sequences: {result.shape}  (cases × bins × features)")
    return result


def build_gaze_static_features(case_paths: list):
    """
    Static gaze features using existing extract_gaze_features().

    Returns
    -------
    X_gaze : np.ndarray, shape (n, d)
    feature_names : list[str]
    """
    import cv2
    from preprocessing.gaze_processing import load_gaze, define_aois, extract_gaze_features
    from config import GAZE_FILE, IMAGE_FILE

    feat_rows = []
    for cp in case_paths:
        gaze_df = load_gaze(os.path.join(cp, GAZE_FILE))
        img = cv2.imread(os.path.join(cp, IMAGE_FILE))
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 512, 512
        aois = define_aois(w, h)
        feats = extract_gaze_features(gaze_df, aois)

        row = [
            feats.get("fixation_count", 0),
            feats.get("mean_fixation_duration", 0),
            feats.get("max_fixation_duration", 0),
            feats.get("scanpath_length", 0),
            feats.get("revisit_rate", 0),
            feats.get("aoi_entropy", 0),
            feats.get("mean_velocity", 0),
            feats.get("std_velocity", 0),
        ]
        # AOI dwell fractions
        dwell = feats.get("dwell_time_per_aoi", {})
        for aoi_name in ["left_lung", "right_lung", "heart", "lower_left", "lower_right", "background"]:
            row.append(dwell.get(aoi_name, 0.0))
        feat_rows.append(row)

    feat_names = [
        "gaze_fixation_count", "gaze_mean_fix_dur", "gaze_max_fix_dur",
        "gaze_scanpath_len", "gaze_revisit_rate", "gaze_aoi_entropy",
        "gaze_mean_velocity", "gaze_std_velocity",
        "gaze_dwell_left_lung", "gaze_dwell_right_lung", "gaze_dwell_heart",
        "gaze_dwell_lower_left", "gaze_dwell_lower_right", "gaze_dwell_background",
    ]

    X = np.array(feat_rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Gaze static features: {X.shape}")
    return X, feat_names
