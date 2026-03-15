"""
classification/image_encoder.py
Image branch feature extraction — deep (ResNet18) or manual fallback.
"""
import os
import numpy as np


def extract_manual_image_features(image_path: str) -> np.ndarray:
    """
    21-dim handcrafted image features.

    Features
    --------
    [0]       mean intensity
    [1]       std intensity
    [2]       skewness
    [3]       kurtosis
    [4-13]    10-bin normalised histogram
    [14]      edge density (mean Sobel gradient)
    [15-18]   quadrant means (TL, TR, BL, BR)
    [19]      horizontal symmetry (|left - right| mean)
    [20]      vertical symmetry (|top - bottom| mean)
    """
    import cv2
    from scipy.stats import skew, kurtosis as kurt

    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(21, dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224)).astype(np.float32) / 255.0
    flat = gray.ravel()

    feat_intensity = [flat.mean(), flat.std(), float(skew(flat)), float(kurt(flat))]

    hist, _ = np.histogram(flat, bins=10, range=(0, 1))
    hist = hist.astype(np.float32) / hist.sum()

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = [float(np.sqrt(sobelx**2 + sobely**2).mean())]

    h, w = gray.shape
    mh, mw = h // 2, w // 2
    quadrant_means = [
        gray[:mh, :mw].mean(),
        gray[:mh, mw:].mean(),
        gray[mh:, :mw].mean(),
        gray[mh:, mw:].mean(),
    ]

    h_sym = [float(np.abs(gray[:, :mw] - gray[:, mw:][:, ::-1]).mean())]
    v_sym = [float(np.abs(gray[:mh, :] - gray[mh:, :][::-1, :]).mean())]

    return np.array(
        feat_intensity + list(hist) + edge_density + quadrant_means + h_sym + v_sym,
        dtype=np.float32,
    )


def extract_deep_image_features(image_path: str):
    """
    512-dim ResNet18 avgpool embedding.
    Returns None if torchvision is unavailable.
    """
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0)  # (1,3,224,224)

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        # Remove FC + avgpool → keep up to layer4
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        with torch.no_grad():
            emb = feature_extractor(x)  # (1,512,1,1)
        return emb.squeeze().numpy()  # (512,)
    except Exception:
        return None


def build_image_features(case_paths: list):
    """
    Build image feature matrix for all cases.

    Returns
    -------
    X_img : np.ndarray, shape (n, d)
    feature_names : list[str]
    """
    from config import IMAGE_FILE

    # Try deep features on first case to decide strategy
    first_img = os.path.join(case_paths[0], IMAGE_FILE)
    test_deep = extract_deep_image_features(first_img)
    use_deep = test_deep is not None

    raw = []
    for cp in case_paths:
        img_path = os.path.join(cp, IMAGE_FILE)
        if use_deep:
            feat = extract_deep_image_features(img_path)
            if feat is None:
                feat = np.zeros(512, dtype=np.float32)
        else:
            feat = extract_manual_image_features(img_path)
        raw.append(feat)

    X = np.stack(raw).astype(np.float32)

    if use_deep:
        from sklearn.decomposition import PCA
        n_components = min(20, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X).astype(np.float32)
        feat_names = [f"img_pca_{i}" for i in range(X.shape[1])]
        source = "ResNet18"
    else:
        feat_names = (
            ["img_mean", "img_std", "img_skew", "img_kurt"]
            + [f"img_hist_{i}" for i in range(10)]
            + ["img_edge_density"]
            + ["img_q_tl", "img_q_tr", "img_q_bl", "img_q_br"]
            + ["img_h_sym", "img_v_sym"]
        )
        source = "manual"

    print(f"Image features: {X.shape} — source: {source}")
    return X, feat_names
