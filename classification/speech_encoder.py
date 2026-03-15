"""
classification/speech_encoder.py
Speech branch — temporal sequences and static features.
"""
import os
import numpy as np


def build_speech_sequence(transcription_df, encoder, n_bins: int = 30,
                          pca_model=None) -> np.ndarray:
    """
    Build temporal speech sequence of shape (n_bins, 16).

    PCA (384→16) must be pre-fitted; pass pca_model.
    If pca_model is None, returns raw 384-dim bins (used for fitting PCA).
    """
    if transcription_df is None or len(transcription_df) == 0:
        if pca_model is not None:
            return np.zeros((n_bins, 16), dtype=np.float32)
        return np.zeros((n_bins, 384), dtype=np.float32)

    # Determine time range from transcription
    if "timestamp_start" in transcription_df.columns:
        t_min = transcription_df["timestamp_start"].min()
        t_max = transcription_df["timestamp_end"].max()
    else:
        t_min, t_max = 0, len(transcription_df)

    if t_max == t_min:
        t_max = t_min + 1e-6
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)

    raw_dim = 384
    result = np.zeros((n_bins, raw_dim), dtype=np.float32)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if "timestamp_start" in transcription_df.columns:
            mask = (transcription_df["timestamp_start"] < hi) & \
                   (transcription_df["timestamp_end"] > lo)
        else:
            # Fall back to row-index-based splitting
            row_lo = int(b * len(transcription_df) / n_bins)
            row_hi = int((b + 1) * len(transcription_df) / n_bins)
            mask = transcription_df.index.isin(range(row_lo, row_hi))

        bin_df = transcription_df[mask]
        if len(bin_df) == 0:
            continue
        texts = bin_df["text"].dropna().tolist()
        if not texts:
            continue
        combined = " ".join(texts)
        # Call the underlying SentenceTransformer model directly (not the wrapper)
        emb = encoder.model.encode([combined])[0]  # 384-dim
        result[b] = emb.astype(np.float32)

    if pca_model is not None:
        # Project each bin: (n_bins, 384) → (n_bins, 16)
        result_pca = pca_model.transform(result).astype(np.float32)
        return result_pca
    return result


def build_all_speech_sequences(case_paths: list, n_bins: int = 30):
    """
    Build (n_cases, n_bins, 16) speech sequence tensor.
    Fits PCA on all non-zero bins across the dataset.
    """
    from preprocessing.speech_processing import SpeechEncoder
    from sklearn.decomposition import PCA
    from config import TRANSCRIPTION_FILE

    enc = SpeechEncoder()

    # Pass 1: collect raw 384-dim embeddings
    raw_sequences = []
    for cp in case_paths:
        trans_df = enc.load_transcription(os.path.join(cp, TRANSCRIPTION_FILE))
        seq = build_speech_sequence(trans_df, enc, n_bins=n_bins, pca_model=None)
        raw_sequences.append(seq)

    # Fit PCA on all non-zero bins
    all_embs = np.concatenate(raw_sequences, axis=0)  # (n_cases*n_bins, 384)
    nonzero_mask = (all_embs != 0).any(axis=1)
    n_nonzero = nonzero_mask.sum()
    n_components = min(16, n_nonzero - 1, 384) if n_nonzero > 1 else 1
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_embs[nonzero_mask])

    # Pass 2: apply PCA
    sequences_pca = []
    for cp in case_paths:
        trans_df = enc.load_transcription(os.path.join(cp, TRANSCRIPTION_FILE))
        seq = build_speech_sequence(trans_df, enc, n_bins=n_bins, pca_model=pca)
        sequences_pca.append(seq)

    result = np.stack(sequences_pca).astype(np.float32)
    print(f"Speech sequences: {result.shape}  (cases × bins × features)")
    return result


def build_speech_static_features(case_paths: list):
    """
    Static speech features: PCA-reduced embedding (10) + keyword counts (4).

    Returns
    -------
    X_speech : np.ndarray, shape (n, 14)
    feature_names : list[str]
    """
    from preprocessing.speech_processing import SpeechEncoder, extract_speech_features
    from sklearn.decomposition import PCA
    from config import TRANSCRIPTION_FILE

    enc = SpeechEncoder()

    embeddings = []
    keyword_rows = []

    for cp in case_paths:
        trans_df = enc.load_transcription(os.path.join(cp, TRANSCRIPTION_FILE))
        emb = enc.encode(trans_df)  # 384-dim
        embeddings.append(emb)

        sf = extract_speech_features(trans_df)
        keyword_rows.append([
            sf.get("anatomy_mentions", 0),
            sf.get("finding_mentions", 0),
            sf.get("negation_count", 0),
            sf.get("uncertainty_count", 0),
        ])

    emb_matrix = np.stack(embeddings).astype(np.float32)
    n_components = min(10, emb_matrix.shape[0] - 1, emb_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    emb_pca = pca.fit_transform(emb_matrix).astype(np.float32)

    kw_matrix = np.array(keyword_rows, dtype=np.float32)
    X = np.concatenate([emb_pca, kw_matrix], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    feat_names = (
        [f"speech_emb_pca_{i}" for i in range(emb_pca.shape[1])]
        + ["speech_anatomy_mentions", "speech_finding_mentions",
           "speech_negation_count", "speech_uncertainty_count"]
    )
    print(f"Speech static features: {X.shape}")
    return X, feat_names
