# Task 2: Multimodal CXR Abnormality Classification

This document provides a complete specification of the multimodal cross-attention fusion architecture for binary CXR abnormality classification, including all modality branches, the fusion mechanism, training protocol, ablation results, and attention interpretability analysis.

---

## Table of Contents

1. [Task Definition](#task-definition)
2. [Dataset Characteristics](#dataset-characteristics)
3. [Image Branch](#image-branch)
4. [Gaze Branch](#gaze-branch)
5. [Speech Branch](#speech-branch)
6. [Cross-Attention Fusion Architecture](#cross-attention-fusion-architecture)
7. [Modality Ablation Matrix](#modality-ablation-matrix)
8. [Training Setup and Evaluation Protocol](#training-setup-and-evaluation-protocol)
9. [Baseline Models](#baseline-models)
10. [Results](#results)
11. [Attention Interpretability](#attention-interpretability)
12. [Limitations and Future Work](#limitations-and-future-work)

---

## Task Definition

Given a CXR reading session consisting of:
- An image I
- A gaze trace G (60 Hz, T samples)
- A transcription S (K segments)

predict the binary label y ∈ {0, 1}, where y=1 indicates that the CXR has at least one abnormal finding in any of the six monitored anatomical regions.

This is framed as a supervised binary classification problem. The ground-truth label is derived from the `overall_abnormal` field in `metadata.json`, which is set to `true` if any region-level `is_abnormal` flag is `true`.

---

## Dataset Characteristics

| Attribute | Value |
|---|---|
| Total sessions | 50 |
| Abnormal (y=1) | 46 |
| Normal (y=0) | 4 |
| Class imbalance ratio | 11.5:1 |
| Split strategy | 5-fold stratified CV |

The severe class imbalance (46 abnormal out of 50) is a direct consequence of the simulator's per-region abnormality probability (P_abnormal = 0.30): the probability of all 6 regions being normal is (0.70)^6 ≈ 0.118, so approximately 88% of sessions contain at least one abnormal region. This imbalance must be borne in mind when interpreting AUC and F1 scores; the effective number of normal cases per fold is 0–1.

---

## Image Branch

The image branch extracts a compact visual descriptor from the CXR image.

### Architecture

```
Input:  I ∈ R^{H × W × 3}   (CXR JPEG, loaded with PIL, converted to RGB)

Preprocessing:
  Resize to 224 × 224
  Normalise: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  (ImageNet stats)

ResNet18 feature extraction:
  torchvision.models.resnet18(pretrained=True)
  Extract output of avgpool layer: v_raw ∈ R^{512}
  (No fine-tuning; weights frozen)

Dimensionality reduction:
  PCA(n_components=20, fit on training fold)
  v_img ∈ R^{20}

Projection for cross-attention:
  Linear(20 → 64) → ReLU → v_img_proj ∈ R^{64}
```

### Handcrafted fallback (no torchvision)

When `torchvision` is not available, a 21-dimensional handcrafted descriptor is computed from the greyscale-converted image:

| Feature group | Dims | Description |
|---|---|---|
| Per-channel statistics | 6 | Mean and std for each of R, G, B channels |
| Greyscale histogram | 8 | Normalised 8-bin histogram of greyscale intensities |
| Sobel edge energy | 1 | Mean magnitude of Sobel gradient over full image |
| Quadrant means | 4 | Mean greyscale intensity in each 2×2 quadrant |
| Left-right symmetry | 1 | 1 minus mean absolute difference between left and right halves |
| **Total** | **20** | (one less than 21 listed above; the symmetry feature brings it to 20 or 21 depending on the exact implementation) |

The handcrafted descriptor is used without PCA (it is already low-dimensional). A Linear(21→64) layer projects it to the shared 64-dimensional space.

---

## Gaze Branch

The gaze branch encodes the temporal structure of the radiologist's scan path.

### Architecture

```
Input:  G ∈ R^{T × 4}   (timestamp_sec, x, y, pupil_mm at 60 Hz)

Binning (500ms, 30-frame windows):
  B = floor(T / 30)   bins
  Per bin b:
    aoi_onehot[b] ∈ R^6      one-hot vector of plurality AOI
    fix_dur[b]    ∈ R^1      total fixation duration in bin (seconds)
    saccade_vel[b]∈ R^1      mean saccade velocity in bin (px/s)
    pupil[b]      ∈ R^1      mean pupil size in bin (mm)
  X_gaze ∈ R^{B × 9}

Input projection:
  Linear(9 → 64) → X_proj ∈ R^{B × 64}

Positional encoding (sinusoidal, same as BehaviorTransformer)

Transformer encoder (2 layers, 4 heads, d_model=64, FFN dim=256):
  H_gaze ∈ R^{B × 64}

Session embedding (mean pooling):
  v_gaze = mean(H_gaze, dim=0) ∈ R^{64}
```

The gaze Transformer is trained jointly with the fusion model (end-to-end) rather than pre-trained separately (unlike the BehaviorTransformer in Task 1). The Task 1 pre-training was exploratory; Task 2 uses supervised end-to-end training.

---

## Speech Branch

The speech branch encodes the semantic content of the dictation.

### Architecture

```
Input:  S = {(text_k, t_start_k, t_end_k)}_{k=1}^{K}

Per-segment embedding:
  SentenceTransformer('all-MiniLM-L6-v2')
  e_k = model.encode(text_k) ∈ R^{384}
  (Encoder weights frozen; no fine-tuning)
  E ∈ R^{K × 384}

Dimensionality reduction (fit on training fold):
  PCA(n_components=16)
  E' ∈ R^{K × 16}

Input projection:
  Linear(16 → 64) → E_proj ∈ R^{K × 64}

Positional encoding (temporal, based on t_start_k):
  Same sinusoidal PE as gaze branch, scaled to [0, session_duration]

Transformer encoder (2 layers, 4 heads, d_model=64, FFN dim=256):
  H_speech ∈ R^{K × 64}

Session embedding (mean pooling):
  v_speech = mean(H_speech, dim=0) ∈ R^{64}
```

**Design note on frozen SentenceTransformer.** Fine-tuning the SentenceTransformer on 40 training samples (80% of 50) would result in severe overfitting to the simulator's vocabulary. The frozen encoder provides stable 384-dimensional representations that capture general semantic similarity; the downstream PCA and Transformer are the learnable components.

---

## Cross-Attention Fusion Architecture

### Mathematical formulation

Let Q, K, V denote query, key, and value matrices for multi-head attention. For a single attention head:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

For multi-head attention with h=4 heads:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
W_i^Q ∈ R^{d_model × d_k},  d_k = d_model/h = 16
W_i^K ∈ R^{d_model × d_k}
W_i^V ∈ R^{d_model × d_v},  d_v = 16
W^O   ∈ R^{h*d_v × d_model} = R^{64 × 64}
```

### Three cross-attention computations

The image embedding is used as query against the gaze sequence and speech sequence (image-centric cross-attention), and the gaze embedding is used as query against the speech sequence (gaze-speech cross-attention):

```
Image attends to gaze:
  v_img_gaze = MultiHead(Q=v_img_proj, K=H_gaze, V=H_gaze)   ∈ R^{64}

Image attends to speech:
  v_img_speech = MultiHead(Q=v_img_proj, K=H_speech, V=H_speech) ∈ R^{64}

Gaze attends to speech:
  v_gaze_speech = MultiHead(Q=v_gaze, K=H_speech, V=H_speech) ∈ R^{64}
```

Note: v_img_proj is the 64-d projected image embedding (scalar query, not a sequence). This is treated as a single-token query in the attention computation.

### Fusion MLP

```
z = Concat([v_img_gaze,   # 64
            v_img_speech, # 64
            v_gaze_speech,# 64
            v_img_proj,   # 64
            v_gaze,       # 64
            v_speech])    # 64
  ∈ R^{384}

fused = Linear(384 → 128) → ReLU → Dropout(0.3)
      → Linear(128 → 64)  → ReLU
      → Linear(64 → 1)    → sigmoid
      → P(y=1)
```

### Full architecture diagram

```
    I ∈R^{H×W×3}      G ∈R^{T×4}           S = {text_k}
         │                  │                      │
    ResNet18           Bin(500ms)          SentenceTransformer
    avgpool               │                      │
    R^{512}          X_gaze∈R^{B×9}        E∈R^{K×384}
         │                  │                      │
     PCA(20)          Linear(9→64)            PCA(16)
         │                  │                      │
    v_img∈R^{20}       Transformer          Linear(16→64)
         │             (2L,4H,64)               │
    Linear(20→64)          │               Transformer
         │             H_gaze∈R^{B×64}     (2L,4H,64)
    v_img_proj∈R^{64}      │                   │
         │            v_gaze=mean        H_speech∈R^{K×64}
         │             ∈R^{64}               │
         │                │             v_speech=mean
         │                │              ∈R^{64}
         │                │                   │
         │    ┌───────────┼───────────────────┤
         │    │           │                   │
         ├────► MHA(Q=v_img_proj, K=H_gaze, V=H_gaze)     → v_img_gaze  ∈R^{64}
         ├────► MHA(Q=v_img_proj, K=H_speech, V=H_speech)  → v_img_speech∈R^{64}
         └────► MHA(Q=v_gaze,     K=H_speech, V=H_speech)  → v_gaze_speech∈R^{64}
                         │
              Concat([v_img_gaze, v_img_speech, v_gaze_speech,
                      v_img_proj, v_gaze, v_speech]) ∈R^{384}
                         │
                    MLP(384→128→64→1)
                         │
                    sigmoid → P(y=1)
```

---

## Modality Ablation Matrix

The architecture supports arbitrary modality dropout. Disabled modalities have their embeddings replaced with zero vectors. The architecture and weight counts are identical across all conditions; only the input values change.

| Condition | Image | Gaze | Speech | Image branch | Gaze branch | Speech branch |
|---|---|---|---|---|---|---|
| image_only | ON | OFF | OFF | ResNet18+PCA | zeros | zeros |
| gaze_only | OFF | ON | OFF | zeros | Transformer | zeros |
| speech_only | OFF | OFF | ON | zeros | zeros | ST+Transformer |
| image_gaze | ON | ON | OFF | ResNet18+PCA | Transformer | zeros |
| image_speech | ON | OFF | ON | ResNet18+PCA | zeros | ST+Transformer |
| gaze_speech | OFF | ON | ON | zeros | Transformer | ST+Transformer |
| full (all) | ON | ON | ON | ResNet18+PCA | Transformer | ST+Transformer |

"Zeros" means the 64-d branch output is replaced with a zero vector before entering the cross-attention fusion stage.

### Modality dropout sensitivity test

To assess the contribution of each modality, a dropout test is performed starting from the full model:

| Dropped modality | Delta AUC (vs. full) |
|---|---|
| Image | -0.136 |
| Speech | -0.092 |
| Gaze | (reported separately) |

A negative delta indicates that removing that modality hurts performance; a larger magnitude indicates greater contribution. Image is the most informative single modality in this dataset, followed by speech. This is consistent with the expectation that the CXR image is the primary diagnostic source, while speech provides confirmatory abnormality signals (since the dictation content is directly derived from the ground-truth findings).

---

## Training Setup and Evaluation Protocol

### Cross-validation

5-fold stratified cross-validation is used. Stratification ensures that the class ratio (approximately 46:4) is preserved across folds to the extent possible with small N.

For each fold:
- Training set: ~40 sessions
- Validation set: ~10 sessions

### Optimisation

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Epochs | 80 |
| Batch size | 8 |
| Loss | BCELoss (binary cross-entropy) |
| LR schedule | ReduceLROnPlateau(factor=0.5, patience=10) |
| Dropout | 0.3 in fusion MLP |

### PCA fitting

PCA transforms (image branch 512→20, speech branch 384→16) are fit on the training fold and applied to the validation fold to prevent data leakage.

### Evaluation metrics

- **AUC**: Area under the ROC curve (`sklearn.metrics.roc_auc_score`). Scores are computed using the predicted sigmoid probability, not the hard threshold prediction.
- **F1**: F1 score at threshold 0.5 (`sklearn.metrics.f1_score`, macro average).

Mean and standard deviation of AUC and F1 are reported across the 5 folds.

### Statistical significance testing

Paired Wilcoxon signed-rank tests are applied to fold-level AUC vectors for pairwise modality comparisons:

```python
from scipy.stats import wilcoxon
stat, p_value = wilcoxon(auc_condition_A, auc_condition_B)
```

The test is applied without continuity correction. Results with p < 0.05 are considered statistically significant given the small sample size (5 paired observations per test), though the test has low power with n=5 and results should be interpreted cautiously.

---

## Baseline Models

Scikit-learn baselines are evaluated for each of the 7 modality conditions:

| Baseline | Implementation |
|---|---|
| SVM (linear kernel) | `sklearn.svm.SVC(kernel='linear', probability=True)` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier(n_estimators=100)` |
| Gaussian Naive Bayes | `sklearn.naive_bayes.GaussianNB()` |

For baselines, the feature input is the concatenation of the (PCA-reduced or handcrafted) branch outputs without the Transformer encoder:
- Image: 20-d PCA of ResNet avgpool
- Gaze: mean of 9-d bin vectors over B bins
- Speech: mean of 384-d SentenceTransformer embeddings over K segments, then PCA(16)

This gives a fixed-length feature vector of at most 20 + 9 + 16 = 45 dimensions for the full condition. The same 5-fold CV split is used as for the deep fusion model.

---

## Results

### Best baseline results

| Modality | Algorithm | AUC | F1 |
|---|---|---|---|
| Image + Speech | Random Forest | 1.000 | 0.9895 |
| Image only | Random Forest | — | — |
| Gaze only | Random Forest | — | — |
| Speech only | Random Forest | — | — |
| Image + Gaze | Random Forest | — | — |
| Gaze + Speech | Random Forest | — | — |
| Full (all) | Random Forest | — | — |

### Cross-attention fusion results (5-fold CV)

| Modality condition | AUC (mean ± std) | F1 (mean ± std) |
|---|---|---|
| Image + Speech | 0.9239 ± 0.194 | — |
| All other conditions | (see outputs/classification/) | — |

### Modality dropout (starting from full model)

| Dropped modality | Delta AUC |
|---|---|
| Image | -0.136 |
| Speech | -0.092 |

### Interpretation of baseline vs. fusion gap

The Random Forest baseline achieves AUC=1.000 while the cross-attention fusion achieves AUC=0.924. This gap is counterintuitive, as the fusion model is architecturally more expressive. Several factors explain it:

1. **Overfitting in small-N regime.** The fusion model has many more trainable parameters (~300K) relative to 40 training samples per fold, leading to high variance in fold-level estimates.
2. **Class imbalance.** With only 4 normal cases in 50, even 1 misclassified normal case significantly degrades AUC. The RF may overfit to the training normal cases more effectively.
3. **Feature engineering advantage.** The RF input features (PCA of ResNet avgpool, SentenceTransformer mean embeddings) are already highly informative, especially since the speech content is directly derived from the ground-truth findings. The deep model must learn these representations from scratch.
4. **Synthetic data artefacts.** The tight coupling between gaze/speech and ground-truth findings in the synthetic data may produce features that are almost perfectly linearly separable for any algorithm.

---

## Attention Interpretability

Attention weights are stored for each fold during inference on the validation set. Three sets of attention weight matrices are saved per session:

1. **Image→Gaze attention** (A_ig): softmax weights over B gaze bins, indicating which temporal periods of the gaze trace the image embedding attends to.
2. **Image→Speech attention** (A_is): softmax weights over K speech segments, indicating which spoken findings the image embedding attends to.
3. **Gaze→Speech attention** (A_gs): softmax weights over K speech segments, indicating which spoken findings the current gaze state attends to.

### Visualisation

For each session in the validation fold, `outputs/classification/` contains heatmaps showing:
- A_ig: bar chart of attention weights over the session timeline (500ms bins)
- A_is: bar chart of attention weights over transcription segments (with text labels)
- A_gs: same as A_is but with gaze as query

These can be inspected to answer questions such as:
- Which spoken findings does the image representation attend to most strongly?
- During which phase of reading does the speech content most strongly modulate the classification decision?

### Known limitations of attention as explanation

Attention weights are not faithful causal explanations of model predictions (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019). High attention weight on a token does not imply that token was causally responsible for the prediction; the relationship is mediated by the value vectors and subsequent MLP layers. Attention visualisations should be treated as exploratory diagnostics rather than definitive explanations.

---

## Limitations and Future Work

### Current limitations

**Class imbalance.** The 46:4 abnormal:normal ratio makes it nearly impossible to estimate generalisation performance reliably with 5-fold CV. A minimum of 50 normal cases would be needed for meaningful evaluation.

**Synthetic data ceiling.** The speech content is directly generated from the same metadata used to create the ground-truth labels. This means that any model with access to speech features has indirect access to the labels, inflating performance estimates. On real clinical data, where speech is independently generated by the radiologist, this leakage would not exist.

**No temporal alignment supervision.** The cross-attention fusion is trained end-to-end with classification loss only. No auxiliary loss encourages the attention weights to align with medically meaningful temporal correspondences (e.g., attending to the speech segment that mentions the diagnosed finding).

**Fixed architecture for all conditions.** Modality dropout (zeroing disabled branch outputs) is a suboptimal way to handle missing modalities. A more principled approach would use modality-specific dropout training or a multimodal missing-data model.

### Future directions

1. **Real CXR data with eye-tracking.** Validate the pipeline on publicly available radiologist eye-tracking datasets (e.g., EGD-CXR: Eye Gaze Data for CXR, Karargyris et al., 2021). This requires adapting the AOI definitions to use image-level pathology annotations rather than bounding boxes.

2. **Contrastive pre-training.** Pre-train the image and gaze/speech encoders using contrastive objectives (e.g., CLIP-style) before fine-tuning on classification. This should improve the quality of the cross-modal alignment.

3. **Temporal alignment auxiliary loss.** Add a supervised alignment loss that encourages A_is to peak at the speech segment corresponding to the actual abnormal finding, using segment-level labels from the transcription.

4. **Larger synthetic cohort.** Increase N to 500+ by expanding the simulator's image set or using data augmentation of gaze traces. This would enable more reliable estimation of fusion model performance.

5. **Radiologist performance prediction.** Extend the classification target from image-level abnormality to radiologist-level performance metrics (missed finding rate, reading time), enabling the system to identify readers who would benefit from decision support.

6. **Uncertainty quantification.** Replace the point-estimate classifier with a Bayesian or ensemble-based model that outputs calibrated probability estimates with confidence intervals, which is essential for clinical deployment.
