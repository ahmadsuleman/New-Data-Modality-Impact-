# Task 1: Multimodal Behaviour Feature Extraction and Clustering

This document provides a detailed specification of every feature dimension in the 46-dimensional behavioural representation, the BehaviorTransformer pre-training setup, the clustering methodology, and the full ablation results.

---

## Table of Contents

1. [Feature Taxonomy](#feature-taxonomy)
2. [Gaze Scalar Features (Dimensions 0–7)](#gaze-scalar-features-dimensions-07)
3. [AOI Transition PCA (Dimensions 8–12)](#aoi-transition-pca-dimensions-812)
4. [AOI Dwell Fractions (Dimensions 13–18)](#aoi-dwell-fractions-dimensions-1318)
5. [Speech Keyword Features (Dimensions 19–22)](#speech-keyword-features-dimensions-1922)
6. [Cross-Modal Alignment Features (Dimensions 23–25)](#cross-modal-alignment-features-dimensions-2325)
7. [Temporal Embeddings (Dimensions 26–35)](#temporal-embeddings-dimensions-2635)
8. [BehaviorTransformer Architecture and Pre-training](#behaviortransformer-architecture-and-pre-training)
9. [Clustering Methodology](#clustering-methodology)
10. [Cluster Naming Heuristics](#cluster-naming-heuristics)
11. [Ablation Study](#ablation-study)

---

## Feature Taxonomy

The following table enumerates all 46 dimensions of the behavioural feature vector. Dimensions are indexed from 0.

| Dim | Group | Name | Unit | Description |
|---|---|---|---|---|
| 0 | Gaze scalar | fixation_count | count | Total number of fixations in session |
| 1 | Gaze scalar | mean_fix_duration | seconds | Mean fixation duration |
| 2 | Gaze scalar | max_fix_duration | seconds | Maximum fixation duration |
| 3 | Gaze scalar | scanpath_length | pixels | Euclidean sum of inter-fixation centroid distances |
| 4 | Gaze scalar | revisit_rate | fraction [0,1] | Fraction of AOIs visited more than once |
| 5 | Gaze scalar | aoi_entropy | nats | Shannon entropy of per-AOI dwell distribution |
| 6 | Gaze scalar | mean_velocity | px/s | Mean gaze velocity across all 60 Hz samples |
| 7 | Gaze scalar | std_velocity | px/s | Std of gaze velocity across all 60 Hz samples |
| 8 | AOI transition | trans_pc_0 | — | 1st PC of row-normalised 6×6 AOI transition matrix |
| 9 | AOI transition | trans_pc_1 | — | 2nd PC |
| 10 | AOI transition | trans_pc_2 | — | 3rd PC |
| 11 | AOI transition | trans_pc_3 | — | 4th PC |
| 12 | AOI transition | trans_pc_4 | — | 5th PC |
| 13 | AOI dwell | dwell_left_lung | fraction [0,1] | Proportional dwell on left lung |
| 14 | AOI dwell | dwell_right_lung | fraction [0,1] | Proportional dwell on right lung |
| 15 | AOI dwell | dwell_heart | fraction [0,1] | Proportional dwell on heart |
| 16 | AOI dwell | dwell_lower_left | fraction [0,1] | Proportional dwell on lower-left zone |
| 17 | AOI dwell | dwell_lower_right | fraction [0,1] | Proportional dwell on lower-right zone |
| 18 | AOI dwell | dwell_background | fraction [0,1] | Proportional dwell outside all AOIs |
| 19 | Speech | anatomy_mentions | count | Number of anatomy keyword occurrences in dictation |
| 20 | Speech | finding_mentions | count | Number of pathology keyword occurrences in dictation |
| 21 | Speech | negation_count | count | Number of negation tokens in dictation |
| 22 | Speech | uncertainty_count | count | Number of hedging/uncertainty tokens in dictation |
| 23 | Cross-modal | gaze_to_speech_lag | seconds | Mean lag from AOI fixation to corresponding dictation onset |
| 24 | Cross-modal | revisits_before_mention | count | Mean number of prior fixations on an AOI before it is mentioned |
| 25 | Cross-modal | mentioned_aoi_dwell_fraction | fraction [0,1] | Fraction of total dwell on AOIs that are mentioned |
| 26–35 | Temporal | temporal_emb_0 … temporal_emb_9 | — | PCA-reduced BehaviorTransformer [CLS] embedding |

All features are z-score standardised (zero mean, unit variance computed across the N=50 session cohort) before any downstream processing.

---

## Gaze Scalar Features (Dimensions 0–7)

### Fixation segmentation

Let x = (x_1, ..., x_T) and y = (y_1, ..., y_T) be the gaze coordinate sequences sampled at fs = 60 Hz. Define instantaneous velocity:

```
v_t = sqrt((x_{t+1} - x_t)^2 + (y_{t+1} - y_t)^2) * fs    [pixels/second]
```

A sample is classified as a fixation sample if v_t < theta_fix = 30 px/frame. Consecutive fixation samples are merged into fixation events. A fixation event is valid if its duration is at least 80 ms (i.e., at least ceil(0.08 * 60) = 5 consecutive fixation samples).

Let N_fix denote the number of valid fixations. For each fixation i with samples [s_i, e_i]:

```
duration_i = (e_i - s_i + 1) / fs
(cx_i, cy_i) = (mean x over [s_i, e_i], mean y over [s_i, e_i])
aoi_i = argmin_a dist((cx_i, cy_i), centroid(box_a))    if min distance < half-diagonal of box_a
        background                                         otherwise
```

### Feature definitions

**fixation_count** (dim 0):

```
f_0 = N_fix
```

**mean_fix_duration** (dim 1):

```
f_1 = (1 / N_fix) * sum_{i=1}^{N_fix} duration_i
```

**max_fix_duration** (dim 2):

```
f_2 = max_{i=1..N_fix} duration_i
```

**scanpath_length** (dim 3):

The total path length connecting successive fixation centroids:

```
f_3 = sum_{i=1}^{N_fix - 1} sqrt((cx_{i+1} - cx_i)^2 + (cy_{i+1} - cy_i)^2)
```

**revisit_rate** (dim 4):

Let A_visited = {aoi_i : i = 1..N_fix, aoi_i != background} be the multiset of visited AOIs. Let A_unique = set(A_visited). Revisit rate is the fraction of unique AOIs that were visited more than once:

```
f_4 = |{a in A_unique : count(a in A_visited) > 1}| / |A_unique|
```

**aoi_entropy** (dim 5):

Let d_a = sum of duration_i over all fixations assigned to AOI a. Define p_a = d_a / sum_b d_b. Shannon entropy in nats:

```
f_5 = -sum_{a : p_a > 0} p_a * ln(p_a)
```

Maximum possible value: ln(6) ≈ 1.79 nats (uniform distribution across 6 AOIs, excluding background).

**mean_velocity** (dim 6):

```
f_6 = (1/T) * sum_{t=1}^{T-1} v_t
```

**std_velocity** (dim 7):

```
f_7 = std({v_t : t = 1..T-1})
```

---

## AOI Transition PCA (Dimensions 8–12)

### Transition matrix construction

For the fixation sequence (aoi_1, aoi_2, ..., aoi_{N_fix}), count transitions:

```
A_{jk} = |{i : aoi_i = j, aoi_{i+1} = k}|    for j, k in {0,...,5}
```

where AOI indices are: 0=right_lung, 1=left_lung, 2=heart, 3=mediastinum, 4=right_lower, 5=left_lower. Background fixations are excluded from the transition count.

Row-normalise to form the transition probability matrix:

```
P_{jk} = A_{jk} / (sum_k A_{jk} + epsilon)    epsilon = 1e-8
```

If a row sums to zero (AOI never visited), it is replaced with a uniform distribution.

Vectorise: p = vec(P) in R^{36} (row-major order).

### PCA compression

Across the N=50 sessions, form the matrix M in R^{50 x 36} where row i is the transition vector p_i. Apply PCA:

```
M_centred = M - mean(M, axis=0)
U, S, Vt = SVD(M_centred)
Z_trans = M_centred @ Vt[:5].T    in R^{50 x 5}
```

The 5-component PCA retains the dominant modes of variation in transition structure. The first principal component typically captures the overall left-lung/right-lung symmetry of transitions; subsequent components capture heart-centric and lower-zone transition patterns.

---

## AOI Dwell Fractions (Dimensions 13–18)

For each session, compute dwell_a as defined above. The 6-dimensional dwell fraction vector satisfies sum_a frac_a = 1.

The ordering (dims 13–18) is: left_lung, right_lung, heart, lower_left, lower_right, background.

**Note on the simplex constraint.** Because the dwell fractions sum to 1, the feature vector lies on a 5-dimensional simplex. This is an implicit constraint that is not enforced during standardisation. The PCA components of this block will capture 5 degrees of freedom, not 6. This is expected and does not cause errors in downstream processing.

---

## Speech Keyword Features (Dimensions 19–22)

### Keyword lists

**Anatomy keywords** (dim 19):
right lung, left lung, heart, mediastinum, hilum, hila, costophrenic, carina, trachea, aorta, diaphragm, pleura, pleural, lobe, lobar, zone

**Finding keywords** (dim 20):
opacity, opacification, infiltrate, consolidation, effusion, atelectasis, cardiomegaly, pneumothorax, pneumonia, mass, nodule, hilar, widened, blunted, air space, airspace, haziness, haziness

**Negation tokens** (dim 21):
no, not, without, clear, normal, unremarkable, negative, absent, free, intact

**Uncertainty tokens** (dim 22):
possible, possibly, probable, probably, likely, question of, cannot exclude, may represent, suspicious, uncertain, equivocal, suggest, concerning

All keyword matching is case-insensitive. Keyword counts are per full session (all transcription segments concatenated).

---

## Cross-Modal Alignment Features (Dimensions 23–25)

### Dimension 23: gaze_to_speech_lag

For each transcription segment k, identify the AOI(s) mentioned in its text using the anatomy keyword list. For each mentioned AOI a:

1. Find the most recent fixation on AOI a before the segment onset:
   ```
   t_last_fix(k, a) = max {t_fix_i : aoi_i = a, t_fix_i + duration_i/2 < timestamp_start(k)}
   ```
2. Compute lag:
   ```
   lag(k, a) = timestamp_start(k) - t_last_fix(k, a)
   ```
3. If no prior fixation on AOI a exists, this (k, a) pair is excluded.

The feature value is:

```
f_23 = mean over all valid (k, a) pairs of lag(k, a)
```

If no valid pairs exist (e.g., session with no anatomy mentions), f_23 = 0.

**Interpretation.** A short lag (< 0.5 s) indicates that the radiologist dictates immediately after first fixating a region (efficient strategy). A long lag may indicate that the radiologist returns to the image for confirmation before dictating, or that there is a mismatch between gaze and speech content.

### Dimension 24: revisits_before_mention

For each (k, a) pair where AOI a is mentioned in segment k:

```
revisits(k, a) = |{i : aoi_i = a, t_fix_i + duration_i/2 < timestamp_start(k)}|
```

Feature value:

```
f_24 = mean over all valid (k, a) pairs of revisits(k, a)
```

A high value indicates that the radiologist fixates a region multiple times before verbally acknowledging it, consistent with a confirmatory inspection strategy.

### Dimension 25: mentioned_aoi_dwell_fraction

Let M = set of AOI names mentioned in any transcription segment. This is extracted by matching anatomy keywords to AOI names.

```
f_25 = sum_{a in M} dwell_a / sum_{b in {all AOIs}} dwell_b
```

A high value (close to 1) means the radiologist primarily dwells on AOIs they explicitly mention in dictation, indicating high gaze-speech coupling. A low value suggests that the radiologist fixates regions without commenting on them (or vice versa).

---

## Temporal Embeddings (Dimensions 26–35)

### Input sequence construction

The 60 Hz gaze trace is converted to a sequence of AOI tokens at 500 ms resolution:

1. Divide the session into L = floor(T * dt_bin / T) non-overlapping bins of width 500 ms (30 frames each).
2. For each bin b, collect all fixation samples within the bin. Assign the plurality AOI (mode of {aoi_i : sample_i in bin b}). If no fixation samples are present, assign token "background" (index 6).
3. Output: x = (x_1, ..., x_L) where x_t in {0,...,6}, L = floor(SESSION_DURATION_SEC / 0.5) = 120.

### BehaviorTransformer Architecture and Pre-training

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Input: x ∈ {0,...,6}^L,  L = 120                      │
│                                                         │
│  Token embedding: W_emb ∈ R^{7 × 64}                   │
│  h_t = W_emb[x_t]  ∈ R^{64}                           │
│                                                         │
│  Prepend [CLS] token: h_0 ∈ R^{64} (learned)           │
│                                                         │
│  Positional encoding (sinusoidal):                      │
│  PE_{t,2i}   = sin(t / 10000^{2i/64})                  │
│  PE_{t,2i+1} = cos(t / 10000^{2i/64})                  │
│                                                         │
│  H_0 = [h_0 + PE_0, h_1 + PE_1, ..., h_L + PE_L]      │
│       ∈ R^{(L+1) × 64}                                 │
│                                                         │
│  TransformerEncoderLayer × 2:                           │
│    - nhead = 4, d_k = d_v = 16                          │
│    - dim_feedforward = 256                              │
│    - dropout = 0.1                                      │
│    - norm_first = True (pre-LN)                         │
│                                                         │
│  [CLS] output: c = H_final[0] ∈ R^{64}                 │
│                                                         │
│  Pre-training head:                                     │
│    Linear(64 → 7) applied at masked positions           │
└─────────────────────────────────────────────────────────┘
```

#### Pre-training task: masked AOI prediction

15 % of tokens in the input sequence are randomly selected for masking. A masked token x_t is replaced with a learned [MASK] embedding. The model is trained to predict the original token at each masked position using cross-entropy loss:

```
L_mask = -(1/|M|) * sum_{t in M} log softmax(Linear(H_final[t]))[x_t]
```

where M is the set of masked positions. This task forces the encoder to learn contextual representations of AOI sequences, capturing patterns such as "after fixating the right lung and heart, the radiologist typically moves to the left lung."

#### Training details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam, lr=1e-3 |
| Max epochs | 100 |
| Batch size | All N sessions (full-batch) |
| Mask rate | 0.15 |
| Convergence criterion | No improvement > 10% over best loss for 10 epochs |
| Device | CUDA if available, else CPU |

The full-batch setting is used because N=50 is small enough to fit in GPU memory. With larger N, mini-batch training should be adopted.

#### Fallback

If PyTorch is unavailable or the training loss does not satisfy the convergence criterion within 100 epochs, the temporal embedding falls back to direct PCA of the raw AOI token sequence represented as a one-hot matrix: X_onehot ∈ R^{N × (L * 7)}, PCA-reduced to 10 dimensions.

---

## Clustering Methodology

### Algorithm selection

Three algorithm families are evaluated:

**KMeans.** Minimises within-cluster sum of squared Euclidean distances. Requires pre-specification of k. Sensitive to initialisation; 10 random initialisations are used with k-means++ seeding. Evaluated for k in {2, 3, 4, 5, 6, 7}.

**Agglomerative clustering (Ward linkage).** Hierarchical bottom-up clustering that merges pairs minimising the increase in total within-cluster variance. Does not require multiple random starts. Evaluated for k in {2, 3, 4, 5, 6, 7}.

**HDBSCAN.** Density-based hierarchical clustering that discovers clusters of arbitrary shape and explicitly labels low-density points as noise. Grid search over:
- min_cluster_size in {2, 3, 4, 5}
- min_samples in {1, 2, 3}

HDBSCAN configurations with noise fraction >= 25% are excluded from the candidate set, as they leave too many sessions unlabelled for meaningful analysis.

### Selection criterion

The silhouette score s(i) for a point i is:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where a(i) is the mean intra-cluster distance and b(i) is the mean distance to the nearest other cluster. The overall silhouette score is the mean over all non-noise points:

```
S = (1/|non-noise|) * sum_{i : label_i != -1} s(i)
```

The configuration maximising S is selected. For HDBSCAN, the noise points (label = -1) are excluded from the silhouette computation.

---

## Cluster Naming Heuristics

After clustering, each cluster is characterised by the z-scores of its member sessions relative to the full cohort. Cluster names are assigned by matching z-score profiles to a lookup table:

| Cluster name | Elevated features (z > 0.5) | Suppressed features (z < -0.5) |
|---|---|---|
| Systematic Scanner | aoi_entropy, fixation_count, scanpath_length, revisit_rate | — |
| Focused Inspector | dwell_right_lung, dwell_left_lung | aoi_entropy, revisit_rate, scanpath_length |
| Uncertain Revisitor | revisit_rate, revisits_before_mention, uncertainty_count | — |
| Rule-Out Strategist | negation_count, aoi_entropy | finding_mentions |
| Hypothesis-Driven | finding_mentions, mentioned_aoi_dwell_fraction | aoi_entropy |
| Rapid Scanner | mean_velocity, std_velocity | mean_fix_duration, fixation_count |
| Speech-Guided Viewer | anatomy_mentions, gaze_to_speech_lag | revisit_rate |

The matching procedure:
1. Compute per-cluster z-scores for each of the 46 features.
2. For each named strategy, compute the match score as the sum of z-scores for elevated features minus the sum for suppressed features.
3. Assign the name with the highest match score to each cluster.

If two clusters would receive the same name (possible in k=2 solutions), the second cluster receives a generic label "Alternative Strategy".

---

## Ablation Study

Six ablation conditions are evaluated by progressively adding feature groups to the clustering input:

| Condition | Feature groups | Total dims |
|---|---|---|
| gaze_behavioral_only | Gaze scalars | 8 |
| gaze_with_transitions | + AOI transition PCA | 13 |
| gaze_and_dwell | + AOI dwell fractions | 19 |
| add_speech | + Speech keywords | 23 |
| add_crossmodal | + Cross-modal alignment | 26 |
| full_multimodal | + Temporal embeddings | 36 (→46 with full config) |

### Results (N=50 sessions)

| Condition | Best algorithm | k | Silhouette score |
|---|---|---|---|
| gaze_behavioral_only | KMeans | 2 | 0.3047 |
| gaze_with_transitions | KMeans | 2 | — |
| gaze_and_dwell | KMeans | 2 | — |
| add_speech | KMeans | 2 | — |
| add_crossmodal | KMeans | 2 | — |
| full_multimodal | HDBSCAN | 2 | 0.1129 |

**Interpretation.** The gaze-only silhouette (0.3047) is substantially higher than the full multimodal score (0.1129). This is a common phenomenon in high-dimensional feature fusion: additional feature dimensions introduce noise that dilutes the cluster structure present in the lower-dimensional space. The HDBSCAN winner in the full multimodal condition reflects the algorithm's ability to find dense regions in high-dimensional space, but the reduced silhouette score indicates that the clusters are less well-separated.

Several interpretations are possible:
1. The temporal embedding (10 dims) adds noise to a small cohort where the self-supervised pretraining may be underfit.
2. The cross-modal alignment features (3 dims) are linearly correlated with gaze scalar features, adding redundant dimensions that dilute cluster structure without adding new information.
3. The true behavioural structure in a simulated dataset is best captured by gaze kinematics alone, since the speech and cross-modal features are deterministically derived from the same gaze-conditional generative process.

Future work should evaluate multimodal fusion on real radiologist data where speech and gaze provide genuinely complementary information about diagnostic strategy.

### Cluster characterisation (full_multimodal, k=2)

Based on the z-score naming heuristic applied to the two HDBSCAN clusters:

| Cluster | Name | Approximate N | Dominant features |
|---|---|---|---|
| 0 | Systematic Scanner | ~26 | High entropy, long scanpath, many revisits |
| 1 | Focused Inspector | ~24 | Concentrated dwell on lung fields, fewer revisits |

These two archetypes are consistent with the expert/novice dichotomy described in the radiology eye-tracking literature, where systematic scanning is associated with higher detection rates for subtle findings, and focused inspection is associated with faster reading times but potentially higher miss rates for peripheral pathology.
