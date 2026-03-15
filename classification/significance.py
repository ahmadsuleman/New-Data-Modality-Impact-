"""
classification/significance.py
Statistical significance testing for modality contributions.
"""
import numpy as np
from scipy import stats


def paired_significance_tests(ablation_results: list):
    """
    Paired t-tests on per-fold AUC scores between modality conditions.

    Comparisons
    -----------
    - Image only  vs  Image + Gaze          (effect of gaze)
    - Image only  vs  Image + Speech        (effect of speech)
    - Image only  vs  Image + Gaze + Speech (effect of full)
    - Gaze+Speech vs  Image + Gaze + Speech (effect of image on top of bio)
    - Image+Gaze  vs  Image + Gaze + Speech (marginal benefit of speech)
    - Image+Speech vs  Image + Gaze + Speech (marginal benefit of gaze)

    Prints and returns a list of result dicts.
    """
    label_to_res = {r["label"]: r for r in ablation_results}

    comparisons = [
        ("Image only",            "Image + Gaze",           "Effect of gaze on image"),
        ("Image only",            "Image + Speech",         "Effect of speech on image"),
        ("Image only",            "Image + Gaze + Speech",  "Effect of full fusion vs image"),
        ("Gaze only",             "Image + Gaze",           "Effect of image on gaze"),
        ("Speech only",           "Image + Speech",         "Effect of image on speech"),
        ("Gaze + Speech",         "Image + Gaze + Speech",  "Effect of adding image to behavioral"),
        ("Image + Gaze",          "Image + Gaze + Speech",  "Marginal effect of speech"),
        ("Image + Speech",        "Image + Gaze + Speech",  "Marginal effect of gaze"),
    ]

    results = []
    print("\n" + "=" * 75)
    print("Paired Significance Tests (per-fold AUC, paired t-test)")
    print("=" * 75)
    print(f"{'Comparison':<45} {'ΔAUC':>8}  {'p-value':>9}  {'Sig?':>5}")
    print("-" * 75)

    for base_lbl, alt_lbl, description in comparisons:
        if base_lbl not in label_to_res or alt_lbl not in label_to_res:
            continue
        auc_base = np.array(label_to_res[base_lbl]["per_fold_auc"])
        auc_alt  = np.array(label_to_res[alt_lbl]["per_fold_auc"])

        if len(auc_base) < 2 or len(auc_alt) < 2:
            continue

        delta = float(auc_alt.mean() - auc_base.mean())
        t_stat, p_val = stats.ttest_rel(auc_alt, auc_base)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
              ("*" if p_val < 0.05 else "ns"))

        row = {
            "description": description,
            "base": base_lbl,
            "alt": alt_lbl,
            "delta_auc": delta,
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        }
        results.append(row)
        print(f"{description:<45} {delta:>+8.4f}  {p_val:>9.4f}  {sig:>5}")

    print("=" * 75)
    print("  ns = p≥0.05,  * p<0.05,  ** p<0.01,  *** p<0.001")
    return results


def modality_dropout_test(all_data: dict, labels: np.ndarray,
                          model_class, base_model_kwargs: dict,
                          full_auc: float, n_folds: int = 5, epochs: int = 100):
    """
    Train full model then zero out one modality at inference to measure dependency.

    Returns list of dicts per dropped modality.
    """
    from classification.trainer import cross_validate

    dropout_conditions = [
        {"drop": "image",  "label": "Drop image",
         "kwargs": dict(base_model_kwargs, use_image=False, use_gaze=True, use_speech=True)},
        {"drop": "gaze",   "label": "Drop gaze",
         "kwargs": dict(base_model_kwargs, use_image=True, use_gaze=False, use_speech=True)},
        {"drop": "speech", "label": "Drop speech",
         "kwargs": dict(base_model_kwargs, use_image=True, use_gaze=True, use_speech=False)},
    ]

    results = []
    print("\n" + "=" * 65)
    print("Modality Dropout Test")
    print("=" * 65)
    print(f"{'Dropped modality':<18} {'AUC without':>12}  {'ΔAUC':>8}  {'Dependency':>12}")
    print("-" * 65)

    for cond in dropout_conditions:
        metrics, pf_auc, _, _, _, _ = cross_validate(
            model_class, cond["kwargs"], all_data, labels,
            n_folds=n_folds, epochs=epochs,
        )
        auc_without = metrics["auc"]
        delta       = full_auc - auc_without
        dependency  = max(0, delta) / max(full_auc, 1e-8)

        row = {
            "dropped":     cond["drop"],
            "auc_without": auc_without,
            "delta":       delta,
            "dependency":  dependency,
        }
        results.append(row)
        print(f"{cond['label']:<18} {auc_without:>12.4f}  {-delta:>+8.4f}  "
              f"{dependency:>12.4f}")

    print("=" * 65)
    return results
