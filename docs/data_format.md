# Dataset Format Specification

This document defines the directory structure, file schemas, coordinate systems, and data types for all files produced by the simulator and consumed by the downstream tasks.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Input Data: CXR Images](#input-data-cxr-images)
3. [gaze.csv Schema](#gazecsv-schema)
4. [transcription.csv Schema](#transcriptioncsv-schema)
5. [metadata.json Schema](#metadatajson-schema)
6. [audio.wav Specification](#audiowav-specification)
7. [Coordinate System](#coordinate-system)
8. [AOI Name Enumeration](#aoi-name-enumeration)
9. [Session Identifier Convention](#session-identifier-convention)
10. [Encoding and Locale Notes](#encoding-and-locale-notes)

---

## Directory Structure

```
generated_dataset/
└── session_XXXX/           XXXX = zero-padded 4-digit integer, 0000–0099
    ├── gaze.csv
    ├── transcription.csv
    ├── audio.wav
    └── metadata.json
```

One subdirectory is created per CXR image. The session index corresponds to the alphabetically sorted position of the source image file within `cxr_images/`. Session `session_0000` corresponds to the first image, `session_0001` to the second, and so on.

If the simulator is re-run with a different set of images, existing session directories are overwritten without warning.

---

## Input Data: CXR Images

| Property | Specification |
|---|---|
| Location | `cxr_images/` |
| File formats | JPEG (`.jpg`, `.jpeg`); case-insensitive |
| Colour space | RGB or greyscale (handled by PIL; greyscale is converted to RGB) |
| Resolution | No constraint; typical CXR dimensions are 1024×1024 to 3000×2500 |
| Naming | Arbitrary; files are sorted alphabetically to assign session indices |
| Minimum count | 1 |

Images are not modified by the simulator. Downstream tasks load images independently.

---

## gaze.csv Schema

The gaze trace records simulated fixation and saccade events at 60 Hz.

### Column definitions

| Column name | dtype | Unit | Null? | Description |
|---|---|---|---|---|
| `timestamp_sec` | float64 | seconds | No | Time since session start. First value is 0.0. Step size is 1/60 ≈ 0.01667 s. |
| `x` | float64 | pixels | No | Horizontal gaze coordinate. Increases leftward from origin. Clipped to [0, image_width]. |
| `y` | float64 | pixels | No | Vertical gaze coordinate. Increases downward from origin. Clipped to [0, image_height]. |
| `pupil_mm` | float64 | millimetres | No | Estimated pupil diameter. Typical range: 2.5–5.5 mm. |

### File properties

| Property | Value |
|---|---|
| Separator | Comma (`,`) |
| Header | Present (first row) |
| Row count | `SESSION_DURATION_SEC * GAZE_SAMPLE_RATE_HZ` (default: 3600) |
| Encoding | UTF-8 |
| Line endings | LF (Unix) |
| Quote character | Double quote `"` (for header only; data rows contain no strings) |
| Float precision | 6 significant figures |

### Example rows

```
timestamp_sec,x,y,pupil_mm
0.000000,512.341234,412.218765,3.512345
0.016667,513.102341,412.876543,3.498712
0.033333,514.234567,413.445678,3.501234
...
```

---

## transcription.csv Schema

The transcription file records timestamped spoken dictation segments.

### Column definitions

| Column name | dtype | Unit | Null? | Description |
|---|---|---|---|---|
| `timestamp_start` | float64 | seconds | No | Time at which this spoken segment begins. Referenced to the same time origin as gaze.csv. |
| `timestamp_end` | float64 | seconds | No | Estimated time at which this spoken segment ends. `timestamp_end > timestamp_start` always. |
| `text` | str | — | No | Full text of the spoken segment. May contain commas (fields are quoted by pandas). |

### File properties

| Property | Value |
|---|---|
| Separator | Comma (`,`) |
| Header | Present (first row) |
| Row count | Typically 8–14 (one opening phrase, 6 region findings, 1 closing impression, 0–1 interim summaries) |
| Encoding | UTF-8 |
| Line endings | LF (Unix) |
| Quote character | Double quote `"` (applied to `text` field when it contains commas) |
| Float precision | 6 significant figures |

### Ordering

Rows are sorted by `timestamp_start` in ascending order. Segments are non-overlapping: `timestamp_start[k+1] >= timestamp_end[k]` for all k.

### Example rows

```
timestamp_start,timestamp_end,text
0.500000,1.250000,Chest radiograph, PA view.
5.634521,6.891234,"Right lung, clear."
9.123456,10.456789,"Left lung, there is an opacity in the left lower lobe."
14.234567,15.012345,"Heart size, normal in size."
19.345678,20.123456,"Mediastinum, within normal limits."
25.456789,26.234567,"Right lower zone, clear costophrenic angle."
30.567890,31.890123,"Left lower zone, blunted costophrenic angle."
55.678901,58.901234,"Impression: abnormal chest radiograph. Findings include left lung opacity and blunted left costophrenic angle."
```

---

## metadata.json Schema

The metadata file records the ground-truth findings used to generate the dictation and the binary classification label.

### Schema (JSON)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["image_path", "session_id", "findings", "overall_abnormal", "abnormal_regions"],
  "properties": {
    "image_path": {
      "type": "string",
      "description": "Relative path from repository root to source CXR JPEG",
      "example": "cxr_images/image_0042.jpg"
    },
    "session_id": {
      "type": "string",
      "pattern": "^session_[0-9]{4}$",
      "description": "Unique session identifier",
      "example": "session_0042"
    },
    "findings": {
      "type": "object",
      "description": "Per-AOI finding record",
      "properties": {
        "right_lung":   { "$ref": "#/definitions/finding" },
        "left_lung":    { "$ref": "#/definitions/finding" },
        "heart":        { "$ref": "#/definitions/finding" },
        "mediastinum":  { "$ref": "#/definitions/finding" },
        "right_lower":  { "$ref": "#/definitions/finding" },
        "left_lower":   { "$ref": "#/definitions/finding" }
      },
      "required": ["right_lung", "left_lung", "heart", "mediastinum", "right_lower", "left_lower"]
    },
    "overall_abnormal": {
      "type": "boolean",
      "description": "True if any AOI has is_abnormal=true. Used as binary classification label."
    },
    "abnormal_regions": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["right_lung", "left_lung", "heart", "mediastinum", "right_lower", "left_lower"]
      },
      "description": "List of AOI names where is_abnormal=true. Empty array if overall_abnormal=false."
    }
  },
  "definitions": {
    "finding": {
      "type": "object",
      "required": ["is_abnormal", "finding"],
      "properties": {
        "is_abnormal": {
          "type": "boolean",
          "description": "Whether this AOI has an abnormal finding in this session"
        },
        "finding": {
          "type": "string",
          "description": "Short finding phrase used in dictation template"
        }
      }
    }
  }
}
```

### Example metadata.json

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

---

## audio.wav Specification

| Property | Value |
|---|---|
| Container | RIFF/WAVE |
| Encoding | PCM (uncompressed) |
| Sample rate | 22050 Hz |
| Bit depth | 16-bit signed integer (little-endian) |
| Channels | Mono (1 channel) |
| Duration | Variable; approximately proportional to total dictation text length |
| Typical duration | 20–45 seconds |
| Typical file size | 880–1980 KB |

The audio file contains the full dictation text spoken as a single continuous utterance. There are no silence gaps between segments at the audio level; timing information is captured in `transcription.csv`.

If `pyttsx3` fails to synthesise audio (e.g., no audio driver on the system), `audio.wav` will not be created. The absence of `audio.wav` is not an error for downstream tasks (Tasks 1 and 2 do not consume `audio.wav` directly; they use `transcription.csv` instead).

---

## Coordinate System

### Image coordinates

All (x, y) coordinate pairs in `gaze.csv` use the standard raster image coordinate system:

```
(0, 0) ─────────────────► x (pixels)
  │                        (increases rightward)
  │
  │
  ▼ y (pixels)
  (increases downward)
```

- Origin: top-left corner of the CXR image
- x range: [0, image_width_pixels]
- y range: [0, image_height_pixels]
- Coordinates are clipped to the valid range; no coordinates fall outside the image boundaries

### Normalised coordinates (internal use only)

AOI bounding boxes in `config.py` are stored in normalised coordinates [0, 1] along both axes. These are converted to pixel coordinates at runtime:

```
x_pixel = x_norm * image_width
y_pixel = y_norm * image_height
```

The `gaze.csv` file always stores pixel coordinates, never normalised. When downstream code resizes the image (e.g., to 224×224 for ResNet18), it must scale the gaze coordinates accordingly:

```
x_scaled = x_pixel * (224 / image_width)
y_scaled = y_pixel * (224 / image_height)
```

---

## AOI Name Enumeration

The six anatomical AOI names used throughout the codebase are fixed strings. They appear in:
- `config.py` AOI bounding box definitions
- `metadata.json` findings keys
- `gaze_sim.py` AOI assignment labels
- Feature extraction code in `main.py`
- Feature extraction code in `main_classification.py`

| AOI name | Clinical region | Approximate image location |
|---|---|---|
| `right_lung` | Right lung (patient's right, image left) | Left half of image, upper two-thirds |
| `left_lung` | Left lung (patient's left, image right) | Right half of image, upper two-thirds |
| `heart` | Cardiac silhouette | Central one-third of image |
| `mediastinum` | Mediastinal structures | Central vertical strip |
| `right_lower` | Right lower lobe and costophrenic angle | Lower-left quadrant of image |
| `left_lower` | Left lower lobe and costophrenic angle | Lower-right quadrant of image |

Note: CXR images are displayed in standard PA (posterior-anterior) projection convention, where the patient's right side appears on the left side of the image. The AOI names follow clinical convention (patient orientation), not image orientation.

---

## Session Identifier Convention

Session identifiers follow the pattern `session_XXXX` where XXXX is a zero-padded 4-digit integer starting at 0000:

| Session ID | Source image (alphabetical rank) |
|---|---|
| session_0000 | 1st image in sorted cxr_images/ listing |
| session_0001 | 2nd image |
| ... | ... |
| session_0099 | 100th image |

The zero-padding ensures correct lexicographic sorting of session directory names. The maximum supported session index is 9999, corresponding to a maximum dataset size of 10000 sessions.

---

## Encoding and Locale Notes

All CSV files use UTF-8 encoding without BOM. Text in the `text` column of `transcription.csv` uses only ASCII characters (the TTS system's vocabulary is English-only). Float values are formatted with a period (`.`) as the decimal separator regardless of the system locale. Files are written using `pandas.DataFrame.to_csv(encoding='utf-8', float_format='%.6f', index=False)` and should be read with the same encoding specification.

JSON files are written with `json.dump(..., indent=2, ensure_ascii=True)` and use standard JSON encoding (RFC 7159). Boolean values are lowercase `true`/`false`.
