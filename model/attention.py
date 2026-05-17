# model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedPhysicochemicalAttention(nn.Module):
    """
    Multi-head self-attention with gated physicochemical bias injection.
    
    The core architectural novelty:
    
        A = softmax( (QK^T / sqrt(d_k)) + alpha * tanh(W_e*P_electro + W_s*P_steric) )
    
    Where:
        alpha  : per-head learnable scalar gate [num_heads]
                 Initialized near zero — model learns how much physics to trust
        W_e    : learnable scalar projection for electrostatic bias
        W_s    : learnable scalar projection for steric bias
        tanh   : bounds the bias to [-1, 1] preventing attention collapse
        
    Weights inherited from ChemBERTa:
        Q, K, V projection matrices  ← transferred from pretrained
        Output projection            ← transferred from pretrained
        
    Randomly initialized (new parameters):
        alpha, W_e, W_s              ← learned from your ncAA data
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        # Standard attention projections — will receive ChemBERTa weights
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # ── Physicochemical gate parameters ──────────────────────────────
        # alpha: per-head gate — how much each head trusts the physics signal
        # Initialized small (0.1) so early training is dominated by
        # learned QK^T signal, physics influence grows gradually
        self.alpha = nn.Parameter(torch.full((num_heads,), 0.1))

        # Scalar projections for each bias matrix
        # Initialized to 1.0 — equal initial weighting
        self.W_e = nn.Parameter(torch.ones(1))   # electrostatic
        self.W_s = nn.Parameter(torch.ones(1))   # steric
        # ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                           # [B, seq_len, embed_dim]
        P_electro: torch.Tensor | None = None,     # [B, seq_len, seq_len]
        P_steric:  torch.Tensor | None = None,     # [B, seq_len, seq_len]
        padding_mask: torch.Tensor | None = None,  # [B, seq_len] bool
    ) -> torch.Tensor:

        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project to Q, K, V and split into heads
        Q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
        K = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # ── Inject physicochemical bias ───────────────────────────────────
        if P_electro is not None and P_steric is not None:
            # Combine bias matrices with learned scalar projections
            # Both P matrices: [B, T, T] → unsqueeze to [B, 1, T, T] for head broadcast
            physics_bias = torch.tanh(
                self.W_e * P_electro.unsqueeze(1) +
                self.W_s * P_steric.unsqueeze(1)
            )  # [B, 1, T, T]

            # Apply per-head gating: alpha [H] → [1, H, 1, 1]
            gated_bias = physics_bias * self.alpha.view(1, H, 1, 1)

            scores = scores + gated_bias
        # ─────────────────────────────────────────────────────────────────

        # Mask padding tokens — set their scores to -inf before softmax
        if padding_mask is not None:
            # padding_mask: True where token is padding
            # Expand to [B, 1, 1, T] for broadcasting over heads and query positions
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)          # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]

        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """
    Full transformer encoder layer:
        GatedPhysicochemicalAttention → Add & Norm → FFN → Add & Norm
    
    Structurally identical to RoBERTa's encoder layer,
    enabling weight transfer from ChemBERTa for all components
    except the physicochemical gate parameters.
    """

    def __init__(
        self,
        embed_dim: int   = 256,
        num_heads: int   = 8,
        ffn_dim: int     = 1024,
        dropout: float   = 0.1,
    ):
        super().__init__()

        self.attention = GatedPhysicochemicalAttention(
            embed_dim, num_heads, dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network — same as RoBERTa's intermediate + output
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),                    # RoBERTa uses GELU, not ReLU
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        P_electro: torch.Tensor | None = None,
        P_steric:  torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Pre-norm architecture (more stable than post-norm for finetuning)
        residual = x
        x = self.norm1(x)
        x = self.attention(x, P_electro, P_steric, padding_mask)
        x = x + residual   # residual connection

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x