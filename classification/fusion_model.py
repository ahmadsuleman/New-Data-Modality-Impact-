"""
classification/fusion_model.py
Cross-attention multimodal fusion model (PyTorch).

Architecture scaled for small-N datasets (N~50):
  - d_model=32 (default), 2 attention heads
  - 1-layer Transformer encoders (reduced from 2)
  - Higher dropout (0.3) to prevent overfitting
  - Fusion MLP: d_model*6 → 64 → d_model
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ── Positional Encoding ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (seq, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ── Temporal branches ────────────────────────────────────────────────────────

class GazeTemporal(nn.Module):
    """Encodes (batch, n_bins, gaze_feat_dim) → (batch, d_model)."""

    def __init__(self, gaze_feat_dim: int, d_model: int = 32):
        super().__init__()
        self.proj = nn.Linear(gaze_feat_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(2, d_model // 16),
            dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # x: (batch, n_bins, feat_dim) → (n_bins, batch, d_model)
        x = self.proj(x).permute(1, 0, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)          # (n_bins, batch, d_model)
        emb = x.mean(dim=0)              # (batch, d_model)
        return emb, x                    # also return full sequence for cross-attention


class SpeechTemporal(nn.Module):
    """Encodes (batch, n_bins, speech_feat_dim) → (batch, d_model)."""

    def __init__(self, speech_feat_dim: int, d_model: int = 32):
        super().__init__()
        self.proj = nn.Linear(speech_feat_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(2, d_model // 16),
            dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.proj(x).permute(1, 0, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        emb = x.mean(dim=0)
        return emb, x


class ImageBranch(nn.Module):
    """Maps static image features → (batch, d_model)."""

    def __init__(self, image_feat_dim: int, d_model: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_feat_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)  # (batch, d_model)


# ── Cross-Attention Fusion ───────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Three pairwise cross-attention blocks:
      a) image_queries_gaze   : Q=image, K=V=gaze sequence
      b) image_queries_speech : Q=image, K=V=speech sequence
      c) gaze_queries_speech  : Q=gaze,  K=V=speech sequence

    Stores attention weights as attributes for interpretability.
    """

    def __init__(self, d_model: int = 32):
        super().__init__()
        n_heads = max(2, d_model // 16)
        self.attn_img_gaze   = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                                      batch_first=False, dropout=0.1)
        self.attn_img_speech = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                                      batch_first=False, dropout=0.1)
        self.attn_gaze_speech = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                                       batch_first=False, dropout=0.1)

        # 3 attended + 3 original embeddings → concat → d_model
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
        )

        # Saved attention weights (set during forward)
        self.w_img_gaze   = None
        self.w_img_speech = None
        self.w_gaze_speech = None

    def forward(self,
                img_emb,          # (batch, d_model)
                gaze_emb,         # (batch, d_model)
                gaze_seq,         # (n_bins, batch, d_model)
                speech_emb,       # (batch, d_model)
                speech_seq,       # (n_bins, batch, d_model)
                ):

        # MHA expects (seq, batch, d_model)
        img_q   = img_emb.unsqueeze(0)    # (1, batch, d_model) — query
        gaze_q  = gaze_emb.unsqueeze(0)   # (1, batch, d_model)

        attended_img_gaze,    w_ig  = self.attn_img_gaze(
            img_q, gaze_seq, gaze_seq)          # Q=img, K=V=gaze_seq
        attended_img_speech,  w_is  = self.attn_img_speech(
            img_q, speech_seq, speech_seq)      # Q=img, K=V=speech_seq
        attended_gaze_speech, w_gs  = self.attn_gaze_speech(
            gaze_q, speech_seq, speech_seq)     # Q=gaze, K=V=speech_seq

        # (1, batch, d_model) → (batch, d_model)
        a_ig  = attended_img_gaze.squeeze(0)
        a_is  = attended_img_speech.squeeze(0)
        a_gs  = attended_gaze_speech.squeeze(0)

        # Store for later analysis
        self.w_img_gaze    = w_ig.detach()
        self.w_img_speech  = w_is.detach()
        self.w_gaze_speech = w_gs.detach()

        fused = torch.cat([img_emb, gaze_emb, speech_emb,
                           a_ig, a_is, a_gs], dim=-1)
        return self.fusion_mlp(fused)  # (batch, d_model)


# ── Full Multimodal Classifier ───────────────────────────────────────────────

class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier with modality flags.

    Parameters
    ----------
    image_feat_dim  : int
    gaze_feat_dim   : int  (per-bin feature count)
    speech_feat_dim : int  (per-bin feature count, after PCA)
    use_image       : bool
    use_gaze        : bool
    use_speech      : bool
    d_model         : int  (hidden dim, default 32)
    """

    def __init__(self,
                 image_feat_dim: int,
                 gaze_feat_dim: int,
                 speech_feat_dim: int,
                 use_image: bool = True,
                 use_gaze: bool = True,
                 use_speech: bool = True,
                 d_model: int = 32):
        super().__init__()
        self.use_image  = use_image
        self.use_gaze   = use_gaze
        self.use_speech = use_speech
        self.d_model    = d_model

        self.image_branch  = ImageBranch(image_feat_dim, d_model)
        self.gaze_branch   = GazeTemporal(gaze_feat_dim, d_model)
        self.speech_branch = SpeechTemporal(speech_feat_dim, d_model)
        self.fusion        = CrossAttentionFusion(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_feat, gaze_seq, speech_seq):
        """
        img_feat   : (batch, image_feat_dim)
        gaze_seq   : (batch, n_bins, gaze_feat_dim)
        speech_seq : (batch, n_bins, speech_feat_dim)
        """
        img_emb    = self.image_branch(img_feat)
        gaze_emb,   gaze_seq_enc   = self.gaze_branch(gaze_seq)
        speech_emb, speech_seq_enc = self.speech_branch(speech_seq)

        # Zero out disabled modalities — same architecture, masked contribution
        if not self.use_image:
            img_emb = torch.zeros_like(img_emb)
        if not self.use_gaze:
            gaze_emb         = torch.zeros_like(gaze_emb)
            gaze_seq_enc     = torch.zeros_like(gaze_seq_enc)
        if not self.use_speech:
            speech_emb       = torch.zeros_like(speech_emb)
            speech_seq_enc   = torch.zeros_like(speech_seq_enc)

        fused = self.fusion(img_emb, gaze_emb, gaze_seq_enc,
                            speech_emb, speech_seq_enc)
        pred = self.classifier(fused).squeeze(-1)  # (batch,)

        attn_weights = {
            "img_gaze":    self.fusion.w_img_gaze,
            "img_speech":  self.fusion.w_img_speech,
            "gaze_speech": self.fusion.w_gaze_speech,
        }
        return pred, attn_weights
