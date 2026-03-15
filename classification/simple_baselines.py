"""
classification/simple_baselines.py
sklearn baseline classifiers for all 5 modality conditions.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

RANDOM_SEED = 42

BASELINE_CONDITIONS = [
    {"label": "Image only",            "use_image": True,  "use_gaze": False, "use_speech": False},
    {"label": "Gaze only",             "use_image": False, "use_gaze": True,  "use_speech": False},
    {"label": "Speech only",           "use_image": False, "use_gaze": False, "use_speech": True},
    {"label": "Image + Gaze",          "use_image": True,  "use_gaze": True,  "use_speech": False},
    {"label": "Image + Speech",        "use_image": True,  "use_gaze": False, "use_speech": True},
    {"label": "Gaze + Speech",         "use_image": False, "use_gaze": True,  "use_speech": True},
    {"label": "Image + Gaze + Speech", "use_image": True,  "use_gaze": True,  "use_speech": True},
]

CLASSIFIERS = {
    "RandomForest":  RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1
    ),
    "SVM-Linear":    SVC(
        kernel="linear", class_weight="balanced",
        probability=True, random_state=RANDOM_SEED
    ),
    "LogisticReg":   LogisticRegression(
        class_weight="balanced", max_iter=1000,
        random_state=RANDOM_SEED, solver="lbfgs"
    ),
}


def _build_feature_matrix(cond: dict,
                           image_features:   np.ndarray,
                           gaze_features:    np.ndarray,
                           speech_features:  np.ndarray,
                           alignment_features: np.ndarray = None) -> np.ndarray:
    parts = []
    if cond["use_image"]:
        parts.append(image_features)
    if cond["use_gaze"]:
        parts.append(gaze_features)
    if cond["use_speech"]:
        parts.append(speech_features)
    if alignment_features is not None and (cond["use_gaze"] and cond["use_speech"]):
        parts.append(alignment_features)
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(image_features), 1))


def _cv_metrics(clf, X, y, cv):
    """Return mean AUC, F1, balanced accuracy via cross-validation."""
    scaler = StandardScaler()
    pipe   = Pipeline([("scaler", scaler), ("clf", clf)])
    auc_scores  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc",
                                  error_score=np.nan)
    f1_scores   = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="f1", error_score=0.0)
    bal_scores  = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="balanced_accuracy", error_score=0.5)
    return {
        "auc":          float(np.nanmean(auc_scores)),
        "auc_std":      float(np.nanstd(auc_scores)),
        "f1":           float(np.nanmean(f1_scores)),
        "balanced_acc": float(np.nanmean(bal_scores)),
    }


def run_simple_baselines(image_features:    np.ndarray,
                         gaze_features:     np.ndarray,
                         speech_features:   np.ndarray,
                         labels:            np.ndarray,
                         alignment_features: np.ndarray = None,
                         n_folds: int = 5):
    """
    Run sklearn baselines for each modality condition.

    Returns
    -------
    results : list[dict] — one per condition, best model selected
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    print("\n" + "=" * 65)
    print("Simple Baseline Results (sklearn)")
    print("=" * 65)
    print(f"{'Condition':<30} {'Best Model':<16} {'AUC':>7}  {'±':>5}  {'F1':>6}  {'Bal.Acc':>8}")
    print("-" * 65)

    results = []
    for cond in BASELINE_CONDITIONS:
        X = _build_feature_matrix(
            cond, image_features, gaze_features,
            speech_features, alignment_features
        )

        best_result = None
        for clf_name, clf in CLASSIFIERS.items():
            try:
                m = _cv_metrics(clf, X, labels, cv)
                m["clf_name"] = clf_name
                if best_result is None or m["auc"] > best_result["auc"]:
                    best_result = m
            except Exception as e:
                print(f"  [warn] {clf_name} failed: {e}")

        if best_result is None:
            best_result = {"clf_name": "N/A", "auc": 0.5, "auc_std": 0.0,
                           "f1": 0.0, "balanced_acc": 0.5}

        row = {"label": cond["label"], **best_result}
        results.append(row)
        print(f"{cond['label']:<30} {best_result['clf_name']:<16} "
              f"{best_result['auc']:>7.4f}  "
              f"±{best_result['auc_std']:>5.3f}  "
              f"{best_result['f1']:>6.4f}  "
              f"{best_result['balanced_acc']:>8.4f}")

    print("=" * 65)
    return results
