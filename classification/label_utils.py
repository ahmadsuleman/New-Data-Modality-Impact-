"""
classification/label_utils.py
Binary labels for abnormal vs normal CXR classification.
"""
import os
import json
import numpy as np

# Pathology keywords that indicate an abnormal finding
PATHOLOGY_KEYWORDS = [
    "nodule", "mass", "effusion", "opacity", "consolidation",
    "pneumothorax", "cardiomegaly", "atelectasis", "infiltrate",
    "edema", "fracture", "widened", "enlarged", "elevated",
    "blunted", "haziness", "pleural", "infiltration", "calcification",
    "prominent", "abnormal", "abnormality", "lesion", "density",
    "fluid", "thickening", "deviation", "hyperinflation",
]


def _count_abnormal_regions(findings: dict) -> int:
    """Count how many regions contain at least one pathology keyword."""
    count = 0
    for region, text in findings.items():
        text_lower = text.lower()
        if any(kw in text_lower for kw in PATHOLOGY_KEYWORDS):
            count += 1
    return count


# Minimum number of abnormal regions required to label a case as abnormal.
# With ABNORMAL_PROB=0.3 per region and 6 regions, threshold=2 yields a
# balanced split (~58% abnormal, ~42% normal) at N=50.
ABNORMAL_THRESHOLD = 2


def _is_abnormal(findings: dict) -> int:
    """Return 1 if >= ABNORMAL_THRESHOLD regions contain pathology keywords."""
    return int(_count_abnormal_regions(findings) >= ABNORMAL_THRESHOLD)


def get_case_paths(dataset_dir: str):
    """Return sorted list of case directory paths (matches label order)."""
    return sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])


def load_labels(dataset_dir: str):
    """
    Load binary labels (0=normal, 1=abnormal) for every case.

    Returns
    -------
    y : np.ndarray, shape (n_cases,)
    """
    case_paths = get_case_paths(dataset_dir)
    labels = []

    first_printed = False
    for cp in case_paths:
        meta_path = os.path.join(cp, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        # Print schema once
        if not first_printed:
            print("=== Sample metadata.json keys ===")
            print("  Keys:", list(meta.keys()))
            print("  Sample findings:")
            for region, text in meta.get("findings", {}).items():
                print(f"    {region}: {text}")
            print()
            first_printed = True

        label = _is_abnormal(meta.get("findings", {}))
        labels.append(label)

    y = np.array(labels, dtype=int)
    n_normal = int((y == 0).sum())
    n_abnormal = int((y == 1).sum())
    print(f"Class distribution — Normal: {n_normal}, Abnormal: {n_abnormal}  "
          f"(total {len(y)}, threshold: >={ABNORMAL_THRESHOLD} abnormal regions)")
    return y
