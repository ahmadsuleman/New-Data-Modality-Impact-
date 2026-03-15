# System Architecture

This document describes the end-to-end data flow of the Multimodal Radiologist Behaviour Analysis Framework, covering input/output shapes, mathematical notation for key operations, and the rationale behind engineering decisions.

---

## Table of Contents

1. [End-to-End Data Flow](#end-to-end-data-flow)
2. [Stage 0: Simulator](#stage-0-simulator)
3. [Stage 1: Feature Extraction and Clustering](#stage-1-feature-extraction-and-clustering)
   - [Gaze Feature Extraction](#gaze-feature-extraction)
   - [AOI Transition Matrix and PCA](#aoi-transition-matrix-and-pca)
   - [AOI Dwell Fractions](#aoi-dwell-fractions)
   - [Speech Feature Extraction](#speech-feature-extraction)
   - [Cross-Modal Alignment Features](#cross-modal-alignment-features)
   - [BehaviorTransformer](#behaviortransformer)
   - [Feature Concatenation](#feature-concatenation)
   - [BehaviorClustering](#behaviorclustering)
4. [Stage 2: Classification](#stage-2-classification)
   - [Image Branch](#image-branch)
   - [Gaze Branch](#gaze-branch)
   - [Speech Branch](#speech-branch)
   - [Cross-Attention Fusion](#cross-attention-fusion)
   - [Training Protocol](#training-protocol)
5. [Shared Utilities](#shared-utilities)

---

## End-to-End Data Flow

```
CXR Images (N=100 JPEG)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0: simulator/generate.py                                 │
│                                                                 │
│  Input:  image I ∈ R^{H×W×3}                                   │
│                                                                 │
│  gaze_sim.py   → gaze trace G ∈ R^{T×4}    (60 Hz, T≈3600)    │
│  speech_sim.py → transcription S (K segments)                  │
│                → audio.wav                                      │
│  metadata     → findings dict F                                 │
│                                                                 │
│  Output per session:  (G, S, I, F)                              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼  generated_dataset/session_XXXX/
        │
        ├─────────────────────────────────────────────────────────┐
        │  STAGE 1: main.py                                        │
        │                                                         │
        │  G → gaze_scalars        ∈ R^8                         │
        │  G → transition_pca      ∈ R^5                         │
        │  G → dwell_fractions     ∈ R^6                         │
        │  S → speech_keywords     ∈ R^4                         │
        │  (G,S) → cross_modal     ∈ R^3                         │
        │  G → BehaviorTransformer → temporal_emb ∈ R^{10}       │
        │                                                         │
        │  concat → feature_vector ∈ R^{46}                      │
        │        → BehaviorClustering → cluster label c           │
        └─────────────────────────────────────────────────────────┘
        │
        └─────────────────────────────────────────────────────────┐
           STAGE 2: main_classification.py                        │
                                                                  │
           I → ResNet18 avgpool → PCA(20) → v_img ∈ R^{20}       │
           G → bin(500ms) → Transformer → v_gaze ∈ R^{64}        │
           S → SentenceTransformer → PCA(16) →                    │
               Transformer → v_speech ∈ R^{64}                    │
                                                                  │
           CrossAttentionFusion(v_img, v_gaze, v_speech)          │
               → fused ∈ R^{64} → sigmoid → P(abnormal)          │
           └─────────────────────────────────────────────────────┘
```

---

## Stage 0: Simulator

See `docs/simulator.md` for full biomechanical model description. This section summarises the I/O contract.

### Module: `simulator/gaze_sim.py`

```
Input:
  image_path : str            — path to CXR JPEG
  config     : SimConfig      — parameters from simulator/config.py

Output:
  gaze_df    : pd.DataFrame   — shape (T, 4)
               columns: timestamp_sec (float64), x (float64),
                        y (float64), pupil_mm (float64)
               T = int(session_duration_sec * 60)   ≈ 3600 rows
```

The 60 Hz sample rate is fixed. `session_duration_sec` defaults to 60 s, giving T = 3600 rows. Coordinates (x, y) are in image pixels, origin at top-left.

### Module: `simulator/speech_sim.py`

```
Input:
  metadata   : dict           — per-region findings
  gaze_df    : pd.DataFrame   — used to derive look-then-speak timestamps

Output:
  transcription_df : pd.DataFrame  — shape (K, 3)
                     columns: timestamp_start (float64, seconds),
                              timestamp_end   (float64, seconds),
                              text            (str)
  audio_path       : str           — path to written WAV file
```

K is the number of spoken segments (one per finding mention plus opening/closing phrases). Typical K is 8–14.

### Module: `simulator/generate.py`

Iterates over `cxr_images/*.{jpg,jpeg}`, calling `gaze_sim.py`, `speech_sim.py`, and the metadata generator for each image. Writes output to `generated_dataset/session_{i:04d}/`.

---

## Stage 1: Feature Extraction and Clustering

### Gaze Feature Extraction

The raw gaze trace is converted to fixations and saccades using a velocity threshold (default 30 px/frame). Consecutive samples below the threshold are merged into fixations.

**Fixation segmentation:**

Let v_t = sqrt((x_{t+1} - x_t)^2 + (y_{t+1} - y_t)^2) * fs be the velocity at sample t (px/s), where fs = 60 Hz. Samples with v_t < θ_fix = 30 px/frame are classified as fixation samples.

Fixation i spans samples [s_i, e_i] with:
- duration_i = (e_i - s_i) / fs  (seconds)
- centroid (cx_i, cy_i) = mean of (x, y) over [s_i, e_i]

**AOI assignment.** Each fixation centroid is assigned to the nearest AOI bounding box centroid. Fixations outside all boxes are assigned to background.

**8-dimensional gaze scalar vector:**

| Index | Feature | Definition |
|---|---|---|
| 0 | fixation_count | N_fix = number of fixations |
| 1 | mean_fix_duration | (1/N_fix) Σ duration_i |
| 2 | max_fix_duration | max_i duration_i |
| 3 | scanpath_length | Σ_{i=1}^{N_fix-1} ‖centroid_{i+1} - centroid_i‖_2 |
| 4 | revisit_rate | fraction of AOIs visited more than once |
| 5 | aoi_entropy | H = -Σ_a p_a log p_a  where p_a = dwell_a / total_dwell |
| 6 | mean_velocity | mean(v_t) over all samples |
| 7 | std_velocity | std(v_t) over all samples |

### AOI Transition Matrix and PCA

Let A ∈ R^{6×6} be the AOI transition count matrix, where A_{ij} = number of times the fixation sequence transitions from AOI i to AOI j. Row-normalise to obtain a stochastic matrix P: P_{ij} = A_{ij} / (Σ_k A_{ik} + ε).

The 36-dimensional vectorised P is stacked across all sessions and PCA is applied. The top 5 principal components are retained, explaining the dominant patterns in transition behaviour across the cohort.

```
P_flat ∈ R^{N×36}  →  PCA(n_components=5)  →  Z_trans ∈ R^{N×5}
```

### AOI Dwell Fractions

For each session, compute the total dwell time per AOI (sum of fixation durations assigned to that AOI). Normalise by total fixation time:

```
dwell_a = Σ_{i : aoi_i = a} duration_i
frac_a  = dwell_a / Σ_{b} dwell_b
```

The 6-dimensional dwell fraction vector (one per AOI: left_lung, right_lung, heart, lower_left, lower_right, background) sums to 1.

### Speech Feature Extraction

Four scalar counts are extracted from the concatenated transcription text:

| Index | Feature | Extraction method |
|---|---|---|
| 0 | anatomy_mentions | Count of anatomy keyword matches (right_lung, left_lung, heart, mediastinum, hilum, costophrenic, …) |
| 1 | finding_mentions | Count of pathology keyword matches (opacity, effusion, cardiomegaly, pneumothorax, …) |
| 2 | negation_count | Count of negation tokens (no, without, clear, normal, unremarkable, …) |
| 3 | uncertainty_count | Count of hedging tokens (possible, likely, cannot exclude, question of, …) |

Sentence embeddings (all-MiniLM-L6-v2, 384-dim) are used in Task 2 only; in Task 1 only the four scalar counts are retained for clustering.

### Cross-Modal Alignment Features

Three features quantify the temporal coupling between gaze and speech:

**Gaze-to-speech lag.**

For each spoken segment s_k with text mentioning AOI a_k, find the most recent fixation on AOI a_k before the segment onset:

```
t_fix(k) = max { t_fix_i : aoi_i = a_k, t_fix_i < timestamp_start(k) }
lag(k)   = timestamp_start(k) - t_fix(k)
```

The feature value is the mean lag across all segments with a matched prior fixation.

**Revisits before mention.**

For each spoken mention of AOI a_k, count the number of distinct fixations on a_k that occurred before the mention:

```
revisits(k) = |{ i : aoi_i = a_k, t_fix_i < timestamp_start(k) }|
```

Feature value: mean revisits across all matched mentions.

**Mentioned-AOI dwell fraction.**

The fraction of total dwell time spent on AOIs that are explicitly mentioned in the dictation:

```
mentioned_AOIs = { a : a mentioned in any transcription segment }
f_mentioned    = Σ_{a ∈ mentioned_AOIs} dwell_a  /  Σ_b dwell_b
```

### BehaviorTransformer

The temporal embedding captures ordered scan-path structure through self-supervised pre-training.

**Input construction.**

The gaze trace is binned into non-overlapping 500 ms windows. Each bin is represented by the plurality AOI among fixations in that window (or background if no fixation). The sequence of AOI indices forms the input: x = (x_1, x_2, ..., x_L) where x_t ∈ {0,...,6}.

**Architecture:**

```
Input tokens:  x ∈ {0,...,6}^L

Token embed:   E_tok ∈ R^{L × d_model},  d_model = 64
               E_tok[t] = W_emb x_t  (learned, dim 7 → 64)

Pos encoding:  PE[t, 2i]   = sin(t / 10000^{2i/d_model})
               PE[t, 2i+1] = cos(t / 10000^{2i/d_model})

Input:         H_0 = E_tok + PE    ∈ R^{L × 64}

Prepend [CLS] token: H_0 ∈ R^{(L+1) × 64}

Encoder layer (×2):
  H' = LayerNorm(H + MultiHeadAttn(H, H, H))
  H  = LayerNorm(H' + FFN(H'))
  FFN: Linear(64→256) → ReLU → Dropout(0.1) → Linear(256→64)
  MultiHeadAttn: n_heads=4, d_k=16, d_v=16

Output [CLS]:  c = H_final[0]  ∈ R^{64}

Classification head (pre-training):
  logits = Linear(64 → 7)  applied to masked positions
```

**Pre-training task.** 15 % of tokens are randomly replaced with a [MASK] token. Cross-entropy loss is computed only on masked positions:

```
L_mask = - (1/|M|) Σ_{t ∈ M} log softmax(logits_t)[x_t]
```

Training stops at epoch e* where L(e) > 0.9 * min_{e'<e} L(e') for 10 consecutive epochs (relative improvement threshold 10 %).

**Downstream extraction.** After pre-training, the [CLS] embedding c ∈ R^{64} is extracted for each session. PCA(n_components=10) is applied across the N-session matrix to yield the 10-dimensional temporal embedding.

### Feature Concatenation

The 46-dimensional feature vector per session is assembled by horizontal concatenation:

```
f = concat([
    gaze_scalars,      # 8
    transition_pca,    # 5
    dwell_fractions,   # 6
    speech_keywords,   # 4
    cross_modal,       # 3
    temporal_emb       # 10
])  ∈ R^{46}
```

All features are z-score standardised (zero mean, unit variance, computed on training set) before clustering and classification.

### BehaviorClustering

```
Input:  F ∈ R^{N×46}   (standardised feature matrix, N=50)

Evaluated algorithms:
  KMeans(k)            for k ∈ {2,3,4,5,6,7}
  AgglomerativeClustering(k, linkage='ward')  for k ∈ {2,3,4,5,6,7}
  HDBSCAN(min_cluster_size=m, min_samples=s)
      for m ∈ {2,3,4,5}, s ∈ {1,2,3}
      constraint: noise_fraction < 0.25

Selection:  argmax_{algorithm, params} silhouette_score(F, labels)

Output:  labels ∈ Z^N,  silhouette ∈ R,  algorithm_name : str
```

---

## Stage 2: Classification

### Image Branch

```
Input:  I ∈ R^{H×W×3}   (CXR JPEG, resized to 224×224)

torchvision path:
  ResNet18(pretrained=True).avgpool   →   v ∈ R^{512}
  PCA(n_components=20)                →   v_img ∈ R^{20}

fallback (no torchvision):
  21 handcrafted features:
    - per-channel mean, std (6)
    - 8-bin greyscale histogram (8)
    - Sobel edge energy (1)
    - quadrant mean intensities (4)
    - left–right intensity symmetry (1)
    - total: v_img ∈ R^{21}
```

### Gaze Branch

```
Input:  G ∈ R^{T×4}   (60 Hz gaze trace)

Binning (500ms windows):
  B = T / (fs * 0.5)   bins
  per bin b:
    aoi_onehot[b]  ∈ R^6    (plurality AOI, one-hot)
    fix_dur[b]     ∈ R^1    (total fixation duration in bin)
    saccade_vel[b] ∈ R^1    (mean saccade velocity)
    pupil[b]       ∈ R^1    (mean pupil size)
  X_gaze ∈ R^{B×9}

Transformer encoder:
  Linear(9 → 64) → Transformer(2 layers, 4 heads, d_model=64) → R^{B×64}
  Mean pool over B → v_gaze ∈ R^{64}
```

### Speech Branch

```
Input:  S — K transcription segments with text strings

Per-segment embedding:
  SentenceTransformer('all-MiniLM-L6-v2') → e_k ∈ R^{384}
  Stack: E ∈ R^{K×384}

Dimensionality reduction:
  PCA(n_components=16) → E' ∈ R^{K×16}

Projection:
  Linear(16 → 64) → Transformer(2 layers, 4 heads, d_model=64) → R^{K×64}
  Mean pool over K → v_speech ∈ R^{64}
```

### Cross-Attention Fusion

The three modality embeddings and their sequences are fused through three pairwise multi-head attention blocks.

Let Q, K, V denote query, key, value matrices. Multi-head attention:

```
MHA(Q, K, V) = Concat(head_1, ..., head_h) W_O

head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

The three cross-attention computations:

```
A_ig = MHA(Q=v_img,   K=X_gaze,    V=X_gaze)    ∈ R^{64}
A_is = MHA(Q=v_img,   K=E'_speech, V=E'_speech) ∈ R^{64}
A_gs = MHA(Q=v_gaze,  K=E'_speech, V=E'_speech) ∈ R^{64}
```

All six 64-d vectors are concatenated and projected:

```
z = Concat([A_ig, A_is, A_gs, v_img_proj, v_gaze, v_speech]) ∈ R^{384}

fused = MLP(z):
  Linear(384 → 128) → ReLU → Dropout(0.3)
  Linear(128 → 64)  → ReLU
  Linear(64 → 1)    → sigmoid → P(abnormal)
```

**Modality dropout.** When a modality is disabled, its branch output is replaced with a zero vector of the appropriate dimension. The architecture is identical for all ablation conditions; only the zero-masking changes.

### Training Protocol

```
Data split:     StratifiedKFold(n_splits=5, shuffle=True)
Optimiser:      Adam(lr=1e-3, weight_decay=1e-4)
Loss:           BCELoss (binary cross-entropy)
Epochs:         80 per fold
Batch size:     8
LR schedule:    ReduceLROnPlateau(factor=0.5, patience=10)
Attention store: attention weights saved per fold for interpretability
Evaluation:     AUC (sklearn.metrics.roc_auc_score),
                F1  (sklearn.metrics.f1_score, threshold=0.5)
Significance:   scipy.stats.wilcoxon (paired, per-fold AUC)
```

---

## Shared Utilities

### AOI Bounding Boxes

AOI bounding boxes are defined in normalised image coordinates [0, 1] in `config.py`:

| AOI | x_min | y_min | x_max | y_max |
|---|---|---|---|---|
| right_lung | 0.05 | 0.10 | 0.45 | 0.75 |
| left_lung | 0.55 | 0.10 | 0.95 | 0.75 |
| heart | 0.35 | 0.30 | 0.65 | 0.70 |
| mediastinum | 0.40 | 0.05 | 0.60 | 0.85 |
| right_lower | 0.05 | 0.70 | 0.45 | 0.95 |
| left_lower | 0.55 | 0.70 | 0.95 | 0.95 |

Pixel coordinates are converted to normalised by dividing by image width and height respectively.

### Reproducibility

All random operations use `numpy.random.seed` and `torch.manual_seed` set from a global `RANDOM_SEED` variable in `config.py`. PCA transforms are fit on the full N-session matrix (no train/test split in Stage 1, which is an unsupervised analysis). In Stage 2, PCA is fit on the training fold only and applied to the validation fold to prevent data leakage.
