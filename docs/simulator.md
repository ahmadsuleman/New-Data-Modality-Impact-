# Simulator: Reading Session Synthesis

This document describes the biomechanical and acoustic models underpinning the synthetic radiologist reading session generator in `simulator/`.

---

## Table of Contents

1. [Overview](#overview)
2. [Gaze Model](#gaze-model)
   - [Session Structure](#session-structure)
   - [Anatomical Region Weighting](#anatomical-region-weighting)
   - [Saccade Model](#saccade-model)
   - [Fixation Model](#fixation-model)
   - [Pupil Dilation Model](#pupil-dilation-model)
   - [Micro-Saccades and Drift](#micro-saccades-and-drift)
3. [Speech Model](#speech-model)
   - [Finding Generation](#finding-generation)
   - [Dictation Templates](#dictation-templates)
   - [Look-Then-Speak Delay](#look-then-speak-delay)
   - [TTS Pipeline](#tts-pipeline)
4. [Metadata Generation](#metadata-generation)
5. [Configurable Parameters](#configurable-parameters)
6. [Output Format Specification](#output-format-specification)
   - [gaze.csv Schema](#gazecsv-schema)
   - [transcription.csv Schema](#transcriptioncsv-schema)
   - [metadata.json Schema](#metadatajson-schema)
   - [audio.wav Specification](#audiowav-specification)
7. [Coordinate System](#coordinate-system)
8. [Limitations](#limitations)
9. [Future Directions](#future-directions)

---

## Overview

The simulator aims to reproduce the qualitative features of expert radiologist eye-tracking data without requiring clinical participants or a controlled laboratory. It is not intended to be a high-fidelity perceptual model; rather, it provides a dataset that exhibits the statistical properties necessary for the downstream feature extraction and clustering tasks to be non-trivial.

The simulator is grounded in empirical findings from the radiology eye-tracking literature:

- Experts spend more time on lung fields and cardiac silhouette than non-experts (Kundel & Nodine, 1983; Drew et al., 2013).
- Reading sessions alternate between global overview saccades (early phase) and local inspection fixations (late phase).
- Mean fixation duration during diagnostic radiology is 250–350 ms (Mello-Thoms et al., 2002).
- Experts revisit previously-fixated regions more often than novices, consistent with a confirmatory checking strategy.
- Spoken dictation lags behind the fixation of the corresponding region by approximately 500 ms (Leong et al., 2017).

---

## Gaze Model

### Session Structure

A reading session is divided into three phases:

| Phase | Duration (default) | Behaviour |
|---|---|---|
| Overview | 0–5 s | Long saccades covering the full image; short fixations; low AOI specificity |
| Systematic inspection | 5–50 s | AOI-weighted fixations; main diagnostic activity |
| Verification | 50–60 s | Re-fixation on flagged regions; increased revisit rate |

Phase boundaries are soft: the transition from overview to inspection is modelled as a linear increase in AOI weight from 0 (uniform) to 1 (fully weighted) over the first 5 seconds.

### Anatomical Region Weighting

Six AOIs are defined as axis-aligned bounding boxes in normalised [0,1] image coordinates. Each AOI has a base sampling weight derived from the clinical prior that lung fields and cardiac silhouette receive disproportionate attention:

| AOI | Base weight |
|---|---|
| right_lung | 0.25 |
| left_lung | 0.25 |
| heart | 0.20 |
| mediastinum | 0.10 |
| right_lower | 0.10 |
| left_lower | 0.10 |

During the inspection phase, the next fixation target AOI is sampled from a categorical distribution with these weights. A "revisit boost" is applied: the weight of any AOI containing an abnormal finding is multiplied by a factor of 1.5 (configurable), increasing the probability of returning to salient regions.

Within the selected AOI, the fixation centroid is sampled from a bivariate Gaussian centred on the AOI centroid with standard deviation equal to one-quarter of the AOI half-width and half-height respectively.

### Saccade Model

Saccades are modelled using a main-sequence approximation (Bahill et al., 1975):

```
amplitude A = ‖p_{fix,i+1} - p_{fix,i}‖_2    (pixels)
duration  D = α * A^β                            (seconds)
```

with parameters α = 0.002, β = 0.8. This gives realistic durations (20–80 ms for 50–500 px amplitudes). Saccade velocity peaks at approximately A / D * 2 (triangular velocity profile approximation).

During a saccade, gaze position is linearly interpolated between source and target fixation centroids:

```
x(t) = x_src + (x_tgt - x_src) * (t - t_sac_start) / D
y(t) = y_src + (y_tgt - y_src) * (t - t_sac_start) / D
```

Gaussian jitter (sigma = 5 px) is added to saccade samples to simulate measurement noise.

### Fixation Model

Fixation duration is sampled from a log-normal distribution:

```
log(duration) ~ N(mu_fix, sigma_fix)
mu_fix    = log(0.25)   — median fixation duration 250 ms
sigma_fix = 0.40        — log-space standard deviation
```

This produces a right-skewed distribution consistent with empirical fixation duration histograms in radiology (mean ≈ 320 ms, range 80–2000 ms).

During a fixation, gaze position drifts slightly around the centroid:

```
x(t) = cx + epsilon_x(t),   epsilon_x ~ N(0, sigma_drift^2)
y(t) = cy + epsilon_y(t),   epsilon_y ~ N(0, sigma_drift^2)
```

where sigma_drift = 15 px and epsilon is independently sampled at each 60 Hz frame (independent drift, not an integrated Brownian motion, for implementation simplicity).

### Pupil Dilation Model

Pupil diameter is modelled as a function of cognitive load. During fixations on AOIs with abnormal findings, cognitive load is elevated, producing pupil dilation:

```
pupil(t) = pupil_baseline + delta_cognitive * I_aoi_abnormal(t)
           + N(0, sigma_pupil)

pupil_baseline   = 3.5  mm
delta_cognitive  = 0.5  mm   (for abnormal AOI fixations)
sigma_pupil      = 0.1  mm   (measurement noise)
```

This is a simplification; real pupil responses have a latency of ~200 ms and a time constant of ~1 s (the pupillary light response and cognitive dilation are conflated here). The model produces plausible values in the range 2.5–5.5 mm.

### Micro-Saccades and Drift

Independent Gaussian jitter (sigma = 15 px) is applied to every gaze sample throughout the session, including fixation samples. This approximates the combined effect of:
- Micro-saccades (involuntary ~0.5 deg saccades occurring at 1–2 Hz)
- Fixational drift (slow conjugate drift at ~0.1 deg/s)
- Eye-tracker measurement noise (typically 0.1–0.5 deg RMS)

The jitter is independent at each sample; no temporal autocorrelation is modelled.

---

## Speech Model

### Finding Generation

For each AOI, a finding is generated by sampling from a finding template bank with probability controlled by the abnormality rate parameter (default 0.30):

```python
is_abnormal = random() < P_abnormal   # P_abnormal = 0.30

if is_abnormal:
    finding = sample(abnormal_findings_for_aoi)
else:
    finding = "clear" or "unremarkable"
```

The finding template banks per AOI include:

| AOI | Normal phrases | Abnormal findings |
|---|---|---|
| right_lung / left_lung | "clear", "unremarkable" | "opacity", "infiltrate", "consolidation", "effusion", "atelectasis" |
| heart | "normal in size" | "cardiomegaly", "enlarged cardiac silhouette" |
| mediastinum | "within normal limits" | "widened", "mediastinal shift" |
| right_lower / left_lower | "clear costophrenic angle" | "blunted costophrenic angle", "pleural effusion" |

### Dictation Templates

Findings are converted to spoken phrases using a template system:

```
"[AOI name], [finding phrase]."

Examples:
  "Right lung, clear."
  "Left lower zone, there is a blunted costophrenic angle."
  "Heart size, cardiomegaly."
```

An opening phrase ("Chest radiograph, PA view.") and closing phrase ("Impression: [summary].") are prepended and appended.

### Look-Then-Speak Delay

A delay of 500 ms is imposed between the time of fixation on an AOI and the start of dictation for that AOI's finding. This is implemented by:

1. Recording the time t_fix at which the gaze trace first dwells on AOI a for ≥ 200 ms.
2. Setting `timestamp_start` for the corresponding speech segment to t_fix + 0.5.
3. Setting `timestamp_end` to `timestamp_start` + estimated speech duration (characters / speaking_rate).

The estimated speaking rate is 150 words per minute (2.5 words/second), which corresponds to approximately 6 characters per second including spaces.

### TTS Pipeline

Synthesised audio is produced by `pyttsx3`, a cross-platform offline TTS library. The full concatenated dictation text is passed to the TTS engine with:

- Voice: default system voice (platform-dependent)
- Speaking rate: 150 words per minute
- Output format: PCM WAV, 22050 Hz sample rate, mono

If `pyttsx3` fails (e.g., no audio driver available), the audio file is omitted and a warning is logged. The `transcription.csv` is always written regardless of TTS success.

---

## Metadata Generation

The `metadata.json` file records the ground-truth findings used to generate the speech:

```json
{
  "image_path": "cxr_images/image_0042.jpg",
  "session_id": "session_0042",
  "findings": {
    "right_lung": {
      "is_abnormal": false,
      "finding": "clear"
    },
    "left_lung": {
      "is_abnormal": true,
      "finding": "opacity"
    },
    "heart": {
      "is_abnormal": false,
      "finding": "normal in size"
    },
    "mediastinum": {
      "is_abnormal": false,
      "finding": "within normal limits"
    },
    "right_lower": {
      "is_abnormal": false,
      "finding": "clear costophrenic angle"
    },
    "left_lower": {
      "is_abnormal": true,
      "finding": "blunted costophrenic angle"
    }
  },
  "overall_abnormal": true,
  "abnormal_regions": ["left_lung", "left_lower"]
}
```

`overall_abnormal` is `true` if any region has `is_abnormal: true`. This field is used as the binary label for Task 2 classification.

---

## Configurable Parameters

All parameters are defined in `simulator/config.py`. The following table lists the most important:

| Parameter | Default | Description |
|---|---|---|
| `SESSION_DURATION_SEC` | 60 | Total simulated reading time in seconds |
| `GAZE_SAMPLE_RATE_HZ` | 60 | Gaze trace sampling frequency |
| `P_ABNORMAL` | 0.30 | Probability of abnormal finding per AOI per session |
| `FIXATION_DURATION_MEAN_LOG` | log(0.25) | Log-space mean of fixation duration distribution |
| `FIXATION_DURATION_STD_LOG` | 0.40 | Log-space std of fixation duration distribution |
| `SACCADE_ALPHA` | 0.002 | Main-sequence amplitude coefficient (seconds/pixel^beta) |
| `SACCADE_BETA` | 0.80 | Main-sequence amplitude exponent |
| `GAZE_JITTER_STD_PX` | 15.0 | Gaussian jitter std for fixation drift (pixels) |
| `PUPIL_BASELINE_MM` | 3.5 | Baseline pupil diameter (mm) |
| `PUPIL_DELTA_COGNITIVE_MM` | 0.5 | Pupil dilation for abnormal AOI fixations (mm) |
| `PUPIL_NOISE_STD_MM` | 0.1 | Pupil measurement noise std (mm) |
| `LOOK_THEN_SPEAK_DELAY_SEC` | 0.5 | Gaze-to-speech delay (seconds) |
| `TTS_WORDS_PER_MIN` | 150 | Speaking rate for duration estimation |
| `OVERVIEW_PHASE_DURATION_SEC` | 5.0 | Duration of initial global overview phase |
| `REVISIT_BOOST_FACTOR` | 1.5 | AOI weight multiplier for abnormal regions |
| `AOI_BOXES` | (see config.py) | Normalised bounding boxes for each AOI |

---

## Output Format Specification

### gaze.csv Schema

| Column | dtype | Units | Description |
|---|---|---|---|
| `timestamp_sec` | float64 | seconds | Time since session start; step = 1/60 |
| `x` | float64 | pixels | Horizontal gaze position; origin top-left |
| `y` | float64 | pixels | Vertical gaze position; origin top-left |
| `pupil_mm` | float64 | mm | Estimated pupil diameter |

- Row count: `SESSION_DURATION_SEC * GAZE_SAMPLE_RATE_HZ` (default 3600)
- No missing values; all coordinates within [0, image_width] × [0, image_height] after clipping
- Header row present; comma-separated; UTF-8 encoding

### transcription.csv Schema

| Column | dtype | Units | Description |
|---|---|---|---|
| `timestamp_start` | float64 | seconds | Start of spoken segment |
| `timestamp_end` | float64 | seconds | End of spoken segment (estimated) |
| `text` | str | — | Spoken text of segment |

- Row count: 1 (opening) + number of AOI findings + 1 (closing impression) = 8–14
- Segments are non-overlapping: `timestamp_start[k+1] >= timestamp_end[k]`
- Header row present; comma-separated; UTF-8 encoding; text field may contain commas (quoted by pandas default)

### metadata.json Schema

```json
{
  "image_path":       "string — relative path to source CXR JPEG",
  "session_id":       "string — unique session identifier (session_XXXX)",
  "findings": {
    "<aoi_name>": {
      "is_abnormal":  "boolean",
      "finding":      "string — finding phrase used in dictation"
    }
  },
  "overall_abnormal": "boolean — true if any AOI is abnormal",
  "abnormal_regions": "array of strings — list of abnormal AOI names"
}
```

AOI names are exactly: `right_lung`, `left_lung`, `heart`, `mediastinum`, `right_lower`, `left_lower`.

### audio.wav Specification

| Property | Value |
|---|---|
| Format | PCM WAV (RIFF/WAVE) |
| Sample rate | 22050 Hz |
| Bit depth | 16-bit signed integer |
| Channels | Mono (1) |
| Duration | Estimated from text length and speaking rate |

---

## Coordinate System

Gaze coordinates (x, y) use the standard image coordinate system:
- Origin (0, 0): top-left corner of the image
- x increases rightward (column index)
- y increases downward (row index)
- Units: pixels in the displayed/loaded image resolution

The image is not resized during simulation; gaze coordinates are in the native resolution of the input JPEG. Downstream consumers that resize the image (e.g., ResNet preprocessing to 224×224) must apply the corresponding coordinate scaling.

AOI bounding boxes in `config.py` are stored in normalised [0, 1] coordinates and converted to pixel coordinates at runtime by multiplying by image width and height.

---

## Limitations

**Stationarity.** The AOI sampling weights are fixed per session. Real radiologists adapt their search strategy dynamically as findings emerge; for instance, a detected pneumothorax will trigger focused re-inspection of the lung apex. The simulator uses a static revisit boost but does not model full dynamic belief updating.

**Cognitive load.** The pupil model is a static approximation. Real cognitive load responses are delayed (~200 ms latency), graded, and modulated by both top-down (task difficulty) and bottom-up (luminance) factors. The simulator conflates all sources of pupil variation into a single Gaussian noise term plus a binary dilation signal.

**Individual variation.** All sessions are generated from the same parameter distribution; there is no modelling of between-radiologist variation in strategy, experience level, or domain knowledge. Adding a "radiologist type" parameter that biases the AOI weights and fixation duration distribution would be a straightforward extension.

**Image content.** The gaze model does not respond to the actual image content. Salient features (consolidations, pneumothorax rim) should attract fixations in a content-driven model. Integrating a saliency map (e.g., from a trained abnormality detector) as additional AOI weighting would improve realism.

**Temporal autocorrelation.** The Gaussian drift model at fixation is memoryless. Real fixational eye movements exhibit temporal autocorrelation (flicker noise) that is not captured here.

**TTS quality.** `pyttsx3` uses system TTS voices, which vary considerably across platforms. For higher-quality synthesis, replace the TTS backend with a neural TTS system (e.g., Coqui TTS, Azure Cognitive Services).

---

## Future Directions

1. **Content-adaptive gaze.** Integrate a pre-trained CXR saliency model (e.g., from a chest X-ray report generation system) to modulate AOI weights based on actual image abnormality probability.

2. **Radiologist persona model.** Parameterise the simulator with a persona vector (experience level, specialty, fatigue state) that biases reading strategy. This would enable simulation of systematic differences between attending radiologists and residents.

3. **Dynamic belief updating.** Implement a Bayesian evidence accumulation model where the probability of revisiting an AOI depends on the current posterior probability of abnormality given prior fixations.

4. **Realistic inter-saccadic interval.** Add a refractory period after each saccade (minimum 100 ms) consistent with the saccadic refractory period observed in human oculomotor research.

5. **Multi-image sessions.** Extend the simulator to generate sessions where the radiologist reads a series of images (prior and current CXR) simultaneously, with gaze switching between images.

6. **Validation against real data.** Quantitatively compare simulated gaze statistics (fixation count, mean duration, scanpath length, AOI dwell distributions) against published normative data from radiologist eye-tracking studies (e.g., Mello-Thoms et al., Drew et al.) using Kolmogorov-Smirnov tests.
