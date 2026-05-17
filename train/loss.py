# train/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnealedInfoNCE(nn.Module):
    """
    InfoNCE contrastive loss with temperature annealing.
    
    Temperature schedule:
        tau(t) = tau_max * exp(-lambda * t)
        
    Starts warm (tau=0.5): loss is forgiving, model learns coarse structure.
    Cools down (tau=0.07): loss becomes discriminative, model learns fine detail.
    
    Why anneal: at initialization, hard negatives (SMARTS-mutated decoys) are
    indistinguishable from positives in random embedding space. A cold temperature
    at step 0 produces exploding gradients. Warm start stabilizes early training.
    """

    def __init__(
        self,
        tau_max: float      = 0.5,
        tau_min: float      = 0.07,
        anneal_steps: int   = 10000,
    ):
        super().__init__()
        self.tau_max      = tau_max
        self.tau_min      = tau_min
        self.anneal_steps = anneal_steps
        self.register_buffer('step', torch.tensor(0, dtype=torch.float32))

    @property
    def tau(self) -> float:
        """Current temperature based on annealing schedule."""
        import math
        lam = math.log(self.tau_max / self.tau_min) / self.anneal_steps
        tau = self.tau_max * math.exp(-lam * self.step.item())
        return max(tau, self.tau_min)  # floor at tau_min

    def forward(
        self,
        z_anchor:    torch.Tensor,   # [B, D] — your ncAA
        z_positive:  torch.Tensor,   # [B, D] — SMILES enumeration variant
        z_hard_neg:  torch.Tensor,   # [B, D] — SMARTS-mutated decoy
        z_bg_neg:    torch.Tensor,   # [B, D] — UniProt canonical baseline
    ) -> tuple[torch.Tensor, dict]:

        tau = self.tau

        # L2 normalize all vectors — cosine similarity via dot product
        z_a  = F.normalize(z_anchor,   dim=-1)
        z_p  = F.normalize(z_positive, dim=-1)
        z_hn = F.normalize(z_hard_neg, dim=-1)
        z_bg = F.normalize(z_bg_neg,   dim=-1)

        # Positive similarity: anchor · positive [B]
        sim_positive = (z_a * z_p).sum(dim=-1) / tau

        # All negative similarities: anchor · each negative [B, 2B]
        # Stack hard negatives and background negatives as denominator
        negatives = torch.cat([z_hn, z_bg], dim=0)  # [2B, D]
        sim_all = torch.matmul(z_a, negatives.T) / tau  # [B, 2B]

        # InfoNCE: -log( exp(sim_pos) / sum(exp(sim_all)) )
        # Numerically stable via logsumexp
        loss = -sim_positive + torch.logsumexp(sim_all, dim=-1)
        loss = loss.mean()

        # Diagnostic metrics — track these during training
        with torch.no_grad():
            alignment   = -sim_positive.mean().item() * tau
            uniformity  = torch.logsumexp(
                torch.matmul(z_a, z_a.T) / tau, dim=-1
            ).mean().item()

        self.step += 1

        metrics = {
            'loss':        loss.item(),
            'tau':         tau,
            'alignment':   alignment,    # Should decrease during training
            'uniformity':  uniformity,   # Should decrease during training
            'step':        self.step.item(),
        }

        return loss, metrics