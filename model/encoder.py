# model/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from model.attention import TransformerEncoderLayer


class AttentionWeightedPooling(nn.Module):
    """
    Compresses [B, T, D] → [B, D] using a learned query vector.
    
    Superior to mean/max pooling because:
    - Learns which token positions carry pharmacophore information
    - Different molecules will weight different positions
    - Interpretable: high attention weight = pharmacophore-relevant substructure
    
    Mathematically:
        weights = softmax( h_i · w_q )     w_q is learned [D] vector
        z = sum_i( weights_i * h_i )
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, D]
        padding_mask: torch.Tensor | None = None,  # [B, T] True=padding
    ) -> torch.Tensor:

        scores = self.query(x).squeeze(-1)  # [B, T]

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [B, T]
        pooled  = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        return pooled


class NcAATransformerEncoder(nn.Module):
    """
    The full encoder for molecular fingerprint generation.
    
    Architecture:
        Embedding(788, 256) + PositionalEncoding
        → 6 × TransformerEncoderLayer (GatedPhysicochemicalAttention)
        → AttentionWeightedPooling
        → ProjectionHead(256 → 128 → 64)
        
    The projection head maps to the contrastive learning space.
    Following SimCLR: the projection head is used ONLY during training.
    At inference/screening time, use the pooled 256-dim vector, not the 64-dim projection.
    
    Weight transfer from ChemBERTa-2:
        ✅ Token embeddings (vocab extended, new tokens randomly init)
        ✅ Q, K, V, output projections in each attention layer
        ✅ FFN weights (intermediate + output)
        ✅ LayerNorm parameters
        ❌ alpha, W_e, W_s (new params — randomly initialized)
        ❌ Pooling query vector (new — randomly initialized)
        ❌ Projection head (new — randomly initialized)
    """

    def __init__(
        self,
        vocab_size: int   = 788,    # From your tokenizer output
        embed_dim: int    = 256,
        num_heads: int    = 8,
        num_layers: int   = 6,
        ffn_dim: int      = 1024,
        max_seq_len: int  = 128,
        dropout: float    = 0.1,
        pad_token_id: int = 1,
        projection_dim: int = 64,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.embed_dim    = embed_dim

        # Token embedding — rows 0:591 will be initialized from ChemBERTa
        # Rows 591:788 (your new ncAA tokens) randomly initialized
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )

        # Learnable positional encoding — same as RoBERTa
        self.pos_embedding = nn.Embedding(max_seq_len + 2, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_input = nn.LayerNorm(embed_dim)

        # 6 transformer layers with physicochemical attention
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Pharmacophore pooling
        self.pooling = AttentionWeightedPooling(embed_dim)

        # Projection head for contrastive learning
        # Only used during training — discarded at inference
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),          # ← new — prevents projection head memorization
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),          # ← new
            nn.Linear(128, projection_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for new parameters."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)

    def forward(
        self,
        token_ids: torch.Tensor,           # [B, T]
        P_electro: torch.Tensor | None = None,  # [B, T, T]
        P_steric:  torch.Tensor | None = None,  # [B, T, T]
        return_projection: bool = True,
    ) -> dict[str, torch.Tensor]:

        B, T = token_ids.shape

        # Build padding mask — True where token is <pad>
        padding_mask = (token_ids == self.pad_token_id)  # [B, T]

        # Embeddings
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.embedding(token_ids) + self.pos_embedding(positions)
        x = self.norm_input(self.dropout(x))

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, P_electro, P_steric, padding_mask)

        # Pool to molecular fingerprint
        fingerprint = self.pooling(x, padding_mask)  # [B, embed_dim]

        output = {'fingerprint': fingerprint}

        if return_projection:
            output['projection'] = self.projection_head(fingerprint)  # [B, 64]

        return output

    @torch.no_grad()
    def encode(self, token_ids, P_electro=None, P_steric=None) -> torch.Tensor:
        """Inference mode — returns fingerprint only, no projection head."""
        self.eval()
        return self.forward(
            token_ids, P_electro, P_steric, return_projection=False
        )['fingerprint']