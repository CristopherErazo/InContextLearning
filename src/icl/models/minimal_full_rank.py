import math
import torch
import torch.nn as nn


class EmbeddingModule(nn.Module):
  def __init__(self, vocab_size, seq_len, d_model):
    super().__init__()
    self.E = nn.Embedding(vocab_size, d_model)
    self.P = nn.Embedding(seq_len, d_model)
    self.d_model = d_model

  def forward(self, x):
    positions = torch.arange(x.size(1), device=x.device)
    e = self.E(x)
    p = self.P(positions).unsqueeze(0).expand(x.size(0), -1, -1)
    return e, p

class FullRankAttentionLayer(nn.Module):

  def __init__(self, d_model, dropout=0.0, lin_attn=False):
    super().__init__()
    self.WQK = nn.Linear(d_model, d_model, bias=False)
    self.WOV = nn.Linear(d_model, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.lin_attn = lin_attn
    self.d_model = d_model

  def forward(self, Q, K, V, mask):
    """Q shape: (B, L_q, d), K shape: (B, L_k, d), V shape: (B, L_k, d)

    mask shape: (B, L_q, L_k) or (1, L_q, L_k)
    """
    B, L_q, d = Q.shape
    L_k = mask.shape[-1]

    # S shape: (B, L_q, L_k)
    S = self.WQK(Q) @ K.transpose(-2, -1)

    if self.lin_attn:
      # Scale by sqrt(d) / sqrt(L_k) to maintain unit variance
      scale = math.sqrt(d) / math.sqrt(L_k)
      A = S.masked_fill(~mask, 0.0) * scale
    else:
      A = (S / math.sqrt(d)).masked_fill(~mask, float("-inf")).softmax(
          dim=-1
      )

    A = self.dropout(A)

    # Output projection: Y shape (B, L_q, d)
    Y = self.WOV(A @ V)
    return Y, A, S


class UnembeddingModule(nn.Module):

  def __init__(self, d_model, vocab_size):
    super().__init__()
    self.U = nn.Linear(d_model, vocab_size, bias=False)

  def forward(self, X):
    return self.U(X)


class MinimalTransformer(nn.Module):
  """
  A minimal transformer with two full-rank attention layers and embedding/unembedding modules.
  The attention layers are parameterized by full-rank matrices, 
  and the model supports both next-token prediction and last-token prediction modes 
  using a 'pred_mode' argument. The model also includes methods for initializing parameters and registering buffers for precomputed quantities.
  """

  def __init__(self, args):
    super().__init__()
    self.seq_len = args.seq_len
    self.vocab_size = args.vocab_size
    self.d_model = args.d_model
    self.drop = args.dropout
    self.lin_attn = args.lin_attn
    self.beta = args.beta
    self.sigma_0 = args.sigma_0
    # Prediction mode: 'next' (NTP across sequence) or 'last' (LTP at position L-1)
    self.pred_mode = args.pred_mode 

    self.embed = EmbeddingModule(self.vocab_size, self.seq_len, self.d_model)
    self.attn1 = FullRankAttentionLayer(self.d_model, self.drop, self.lin_attn)
    self.attn2 = FullRankAttentionLayer(self.d_model, self.drop, self.lin_attn)
    self.unembed = UnembeddingModule(self.d_model, self.vocab_size)

  def forward(self, x, mask=None):
    """x shape: (B, L)"""
    B, L = x.shape

    e, p = self.embed(x)  # Shapes: (B, L, d)
    X1, _, _ = self.attn1(p, p, e, mask)  # (B, L, d)

    if self.pred_mode == "last":
      q2 = e[:, -1:, :]  # (B, 1, d)
      mask2 = mask[:, -1:, :]  # (B, 1, L)
      # Layer 2 computes attention ONLY for position L-1 attending to 0..L-1
      X2, _, _ = self.attn2(q2, X1, e, mask2)  # Output: (B, 1, d)
    else:
      # --- Standard Next-Token Prediction ---
      X2, _, _ = self.attn2(e, X1, e, mask)  # Output: (B, L, d)

    # Unembedding: Only projects (B, 1, d) in 'last' mode!
    logits = self.beta * self.unembed(X2)  # (B, 1, V) or (B, L, V)
    return logits


  def full_output(self, x, mask):
    out = {}
    e, p = self.embed(x)
    X1, A1, S1 = self.attn1(p, p, e, mask)
    # X2, A2, S2 = self.attn2(e, X1, e, mask)
    if self.pred_mode == "last":
      q2 = e[:, -1:, :]  # (B, 1, d)
      mask2 = mask[:, -1:, :]  # (B, 1, L)
      X2, A2, S2 = self.attn2(q2, X1, e, mask2)  # Output: (B, 1, d)
    else:
      X2, A2, S2 = self.attn2(e, X1, e, mask)  # Output: (B, L, d)
    out["X1"], out["A1"], out["S1"] = X1, A1, S1
    out["X2"], out["A2"], out["S2"] = X2, A2, S2
    out["logits"] = self.beta * self.unembed(X2)
    return out

  def register_buffers(self, K_trig):
    """Precompute quantities derived from frozen parameters."""
    with torch.no_grad():
      E = self.embed.E.weight  # shape (V, d)
      E_trigg = E[:K_trig].mean(dim=0)  # shape (d,)
      E_nontrigg = E[K_trig:].mean(dim=0)  # shape (d,)
      U = self.unembed.U.weight  # shape (V, d)
      U_nontrigg = U[K_trig:].mean(dim=0)  # shape (d,)

    self.register_buffer("E_trigg", E_trigg)
    self.register_buffer("E_nontrigg", E_nontrigg)
    self.register_buffer("U_nontrigg", U_nontrigg)

  def initialize_model(self):
    """Scales full-rank composite parameters for d -> inf stability."""
    d = self.d_model

    for name, param in self.named_parameters():
      if "embed.E" in name or "embed.P" in name:
        # Embeddings: N(0, 1/d) -> Unit norm ||E||_2^2 ~ 1
        param.data.copy_(torch.randn_like(param) / math.sqrt(d))

      elif "unembed.U" in name:
        # Unembedding: N(0, 1/d)
        param.data.copy_(torch.randn_like(param) / math.sqrt(d))

      elif "WQK" in name or "WOV" in name:
        # Composite d x d matrices: N(0, sigma_0^2 / d) -> Spectral norm ||W||_2 ~ O(1)
        param.data.copy_(
            self.sigma_0 * torch.randn_like(param) / math.sqrt(d)
        )

      # Freeze all parameters by default
      param.requires_grad = False

    # Unfreeze active learning weights
    self.attn1.WQK.weight.requires_grad = True

    self.attn2.WQK.weight.requires_grad = True
    self.attn2.WOV.weight.requires_grad = True

