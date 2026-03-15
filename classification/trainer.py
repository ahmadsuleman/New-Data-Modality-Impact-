"""
classification/trainer.py
Training and evaluation with stratified K-fold cross-validation.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_curve
)

from classification.fusion_model import MultimodalClassifier

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype)


def _compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    f1  = f1_score(y_true, y_pred, zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob) if len(np.unique(y_true)) > 1 \
        else ([0, 1], [0, 1], None)
    return {
        "auc": auc, "f1": f1, "balanced_acc": bal,
        "sensitivity": sens, "specificity": spec,
        "fpr": fpr, "tpr": tpr,
    }


# ── Single-fold training ─────────────────────────────────────────────────────

def train_one_fold(model, train_data, val_data,
                   epochs: int = 100, lr: float = 1e-3):
    """
    Parameters
    ----------
    model      : MultimodalClassifier (fresh instance)
    train_data : dict with keys img, gaze, speech, labels (numpy)
    val_data   : same
    epochs     : max epochs
    lr         : learning rate

    Returns
    -------
    best_state : model state dict
    val_probs  : np.ndarray predicted probabilities on val
    val_labels : np.ndarray
    attn_weights : dict
    history    : dict with loss/auc per epoch
    """
    # Pos weight for class imbalance
    train_labels = train_data["labels"]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )
    criterion = nn.BCELoss(weight=None)  # pos_weight applied via manual scaling

    # Build tensors
    img_tr    = _to_tensor(train_data["img"])
    gaze_tr   = _to_tensor(train_data["gaze"])
    speech_tr = _to_tensor(train_data["speech"])
    y_tr      = _to_tensor(train_labels)

    img_val    = _to_tensor(val_data["img"])
    gaze_val   = _to_tensor(val_data["gaze"])
    speech_val = _to_tensor(val_data["speech"])
    y_val      = _to_tensor(val_data["labels"])

    best_auc   = -1.0
    best_state = None
    patience   = 20
    no_improve = 0
    history    = {"train_loss": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(img_tr, gaze_tr, speech_tr)

        # Manual pos_weight scaling to BCELoss
        pw = pos_weight.squeeze()
        loss = (-pw * y_tr * torch.log(pred + 1e-8)
                - (1 - y_tr) * torch.log(1 - pred + 1e-8)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, val_attn = model(img_val, gaze_val, speech_val)
            val_probs = val_pred.numpy()
            val_lbl   = y_val.numpy().astype(int)
            val_auc   = roc_auc_score(val_lbl, val_probs) \
                        if len(np.unique(val_lbl)) > 1 else 0.5

        scheduler.step(val_auc)
        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(float(val_auc))

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred, final_attn = model(img_val, gaze_val, speech_val)

    # Collect attention weights as numpy
    attn_np = {}
    for k, v in final_attn.items():
        if v is not None:
            attn_np[k] = v.numpy()

    return best_state, val_pred.numpy(), val_data["labels"].astype(int), attn_np, history


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate(model_class, model_kwargs: dict, all_data: dict,
                   labels: np.ndarray, n_folds: int = 5, epochs: int = 100):
    """
    Stratified K-fold cross-validation.

    Returns
    -------
    metrics        : dict  (aggregated metrics)
    per_fold_auc   : list[float]
    all_val_probs  : np.ndarray (full dataset, in fold order)
    all_val_labels : np.ndarray
    all_attn       : list[dict]  per-fold attention weights
    all_histories  : list[dict]
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    n = len(labels)
    all_val_probs  = np.zeros(n, dtype=np.float32)
    all_val_labels = np.zeros(n, dtype=int)
    per_fold_auc   = []
    all_attn       = []
    all_histories  = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), labels)):
        # Scale continuous features (img, static gaze/speech are continuous)
        scaler_img    = StandardScaler()
        scaler_gaze   = StandardScaler()
        scaler_speech = StandardScaler()

        img_tr_raw    = all_data["img"][train_idx]
        img_val_raw   = all_data["img"][val_idx]
        gaze_tr_seq   = all_data["gaze"][train_idx]
        gaze_val_seq  = all_data["gaze"][val_idx]
        sp_tr_seq     = all_data["speech"][train_idx]
        sp_val_seq    = all_data["speech"][val_idx]

        # Scale image features (2D)
        img_tr  = scaler_img.fit_transform(img_tr_raw).astype(np.float32)
        img_val_s = scaler_img.transform(img_val_raw).astype(np.float32)

        # Scale gaze sequences (flatten last 2 dims, scale, reshape)
        n_tr, nb, gf = gaze_tr_seq.shape
        gaze_tr_flat  = gaze_tr_seq.reshape(n_tr, -1)
        gaze_val_flat = gaze_val_seq.reshape(len(val_idx), -1)
        gaze_tr_s  = scaler_gaze.fit_transform(gaze_tr_flat).reshape(n_tr, nb, gf).astype(np.float32)
        gaze_val_s = scaler_gaze.transform(gaze_val_flat).reshape(len(val_idx), nb, gf).astype(np.float32)

        n_tr_s, nb_s, sf_dim = sp_tr_seq.shape
        sp_tr_flat  = sp_tr_seq.reshape(n_tr_s, -1)
        sp_val_flat = sp_val_seq.reshape(len(val_idx), -1)
        sp_tr_s  = scaler_speech.fit_transform(sp_tr_flat).reshape(n_tr_s, nb_s, sf_dim).astype(np.float32)
        sp_val_s = scaler_speech.transform(sp_val_flat).reshape(len(val_idx), nb_s, sf_dim).astype(np.float32)

        train_data = {
            "img": img_tr, "gaze": gaze_tr_s, "speech": sp_tr_s,
            "labels": labels[train_idx].astype(np.float32),
        }
        val_data = {
            "img": img_val_s, "gaze": gaze_val_s, "speech": sp_val_s,
            "labels": labels[val_idx].astype(np.float32),
        }

        model = model_class(**model_kwargs)
        _, vp, vl, attn, hist = train_one_fold(
            model, train_data, val_data, epochs=epochs
        )

        fold_auc = roc_auc_score(vl, vp) if len(np.unique(vl)) > 1 else 0.5
        per_fold_auc.append(fold_auc)
        all_val_probs[val_idx]  = vp
        all_val_labels[val_idx] = vl
        all_attn.append(attn)
        all_histories.append(hist)

    metrics = _compute_metrics(all_val_labels, all_val_probs)
    metrics["per_fold_auc"] = per_fold_auc
    metrics["mean_auc"]     = float(np.mean(per_fold_auc))
    metrics["std_auc"]      = float(np.std(per_fold_auc))
    metrics["all_val_probs"]  = all_val_probs
    metrics["all_val_labels"] = all_val_labels
    return metrics, per_fold_auc, all_val_probs, all_val_labels, all_attn, all_histories


# ── Modality ablation ────────────────────────────────────────────────────────

ABLATION_CONDITIONS = [
    {"label": "Image only",              "use_image": True,  "use_gaze": False, "use_speech": False},
    {"label": "Gaze only",               "use_image": False, "use_gaze": True,  "use_speech": False},
    {"label": "Speech only",             "use_image": False, "use_gaze": False, "use_speech": True},
    {"label": "Image + Gaze",            "use_image": True,  "use_gaze": True,  "use_speech": False},
    {"label": "Image + Speech",          "use_image": True,  "use_gaze": False, "use_speech": True},
    {"label": "Gaze + Speech",           "use_image": False, "use_gaze": True,  "use_speech": True},
    {"label": "Image + Gaze + Speech",   "use_image": True,  "use_gaze": True,  "use_speech": True},
]


def run_modality_ablation(all_data: dict, labels: np.ndarray,
                          base_model_kwargs: dict,
                          n_folds: int = 5, epochs: int = 100):
    """
    Train one cross-validated run per modality condition.
    Returns list of result dicts.
    """
    results = []
    header = f"{'Condition':<30} {'AUC':>7}  {'±':>5}  {'F1':>6}  {'Bal.Acc':>8}  {'Sens':>6}  {'Spec':>6}"
    print("\n" + "=" * 70)
    print("Modality Ablation Results")
    print("=" * 70)
    print(header)
    print("-" * 70)

    for cond in ABLATION_CONDITIONS:
        kwargs = dict(base_model_kwargs)
        kwargs["use_image"]  = cond["use_image"]
        kwargs["use_gaze"]   = cond["use_gaze"]
        kwargs["use_speech"] = cond["use_speech"]

        metrics, pf_auc, vp, vl, attn, hist = cross_validate(
            MultimodalClassifier, kwargs, all_data, labels,
            n_folds=n_folds, epochs=epochs,
        )
        row = {
            "label":       cond["label"],
            "use_image":   cond["use_image"],
            "use_gaze":    cond["use_gaze"],
            "use_speech":  cond["use_speech"],
            "per_fold_auc": pf_auc,
            "all_val_probs":  vp,
            "all_val_labels": vl,
            "attention_weights": attn,
            "histories":   hist,
            **{k: v for k, v in metrics.items()
               if k not in ("per_fold_auc", "all_val_probs", "all_val_labels", "fpr", "tpr")},
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
        }
        results.append(row)
        print(f"{cond['label']:<30} {metrics['auc']:>7.4f}  "
              f"±{metrics['std_auc']:>5.3f}  "
              f"{metrics['f1']:>6.4f}  "
              f"{metrics['balanced_acc']:>8.4f}  "
              f"{metrics['sensitivity']:>6.4f}  "
              f"{metrics['specificity']:>6.4f}")

    print("=" * 70)
    return results
