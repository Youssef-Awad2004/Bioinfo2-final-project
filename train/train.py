# train/trainer.py
import os
import sys
from contextlib import nullcontext

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
from model.encoder import NcAATransformerEncoder
from model.weight_transfer import transfer_chemberta_weights
from train.loss import AnnealedInfoNCE
from model.config import TRAINING_CONFIG, load_model_config


def build_model(model_config: dict) -> NcAATransformerEncoder:
    """Builds model and transfers ChemBERTa weights in one call."""
    model = NcAATransformerEncoder(**model_config)
    model = transfer_chemberta_weights(model)
    return model


def get_parameter_groups(model: NcAATransformerEncoder) -> list[dict]:
    """
    Differential learning rates — critical for finetuning stability.
    
    Pretrained weights: low LR (don't destroy what ChemBERTa learned)
    New parameters:     high LR (these need to learn fast)
    """
    pretrained_params = []
    new_params        = []

    new_param_names = [
        'alpha', 'W_e', 'W_s',      # Physics gate — new
        'pooling',                    # Attention pooling — new
        'projection_head',            # Contrastive head — new
    ]

    for name, param in model.named_parameters():
        is_new = any(new_name in name for new_name in new_param_names)

        # Also treat embedding rows for new tokens as new params
        if 'embedding' in name:
            is_new = True  # Will handle partial LR in custom optimizer

        if is_new:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    return [
        {'params': pretrained_params, 'lr': 1e-5},  # Pretrained: slow
        {'params': new_params,        'lr': 1e-4},  # New: fast
    ]


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss  = 0
    total_steps = 0

    use_amp = torch.cuda.is_available() and str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        def encode_role(role):
            ids = batch[role]['token_ids'].to(device)
            P_e = batch[role]['P_electro'].to(device)
            P_s = batch[role]['P_steric'].to(device)
            return model(ids, P_electro=P_e, P_steric=P_s,
                        return_projection=True)['projection']

        autocast_ctx = torch.autocast('cuda') if use_amp else nullcontext()

        with autocast_ctx:
            z_a  = encode_role('anchor')
            z_p  = encode_role('positive')
            z_hn = encode_role('hard_neg')
            z_bg = encode_role('bg_neg')

        loss, metrics = loss_fn(z_a, z_p, z_hn, z_bg)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if grad_norm > 1.0:
            print(f" Gradient clipped: {grad_norm:.2f} → 1.0")
        scaler.step(optimizer)
        scaler.update()

        total_loss  += metrics['loss']
        total_steps += 1

        if total_steps % 5 == 0:
            print(
                f"  Step {total_steps:3d} | "
                f"loss {metrics['loss']:.4f} | "
                f"tau {metrics['tau']:.3f} | "
                f"align {metrics['alignment']:.4f}"
            )

    return {'avg_loss': total_loss / max(total_steps, 1)}

def collate_triplets(batch):
    """
    Custom collate function.
    Each item from Dataset is a dict of dicts.
    This stacks them into batched tensors.
    """
    def stack_role(role):
        return {
            'token_ids': torch.stack([b[role]['token_ids'] for b in batch]),
            'P_electro': torch.stack([b[role]['P_electro'] for b in batch]),
            'P_steric':  torch.stack([b[role]['P_steric']  for b in batch]),
        }

    return {
        'anchor':   stack_role('anchor'),
        'positive': stack_role('positive'),
        'hard_neg': stack_role('hard_neg'),
        'bg_neg':   stack_role('bg_neg'),
    }

def evaluate(model, dataset, device, label="VAL") -> dict:
    """
    Evaluation metrics for contrastive molecular encoder.
    
    Three metrics:
    
    1. Contrastive Loss — same InfoNCE on held-out triplets
       Direction: should be lower than training loss (no overfitting)
    
    2. Pharmacophore Alignment Score — average cosine similarity
       between anchor and positive pairs
       Direction: should be high (>0.5 after training, >0.8 ideally)
    
    3. Discrimination Score — average cosine similarity between
       anchor and hard negative pairs
       Direction: should be LOW — model should push decoys away
       
    The gap between Alignment and Discrimination is your key metric.
    A large gap means your model correctly separates pharmacophore
    matches from decoys. This is the core scientific claim.
    """
    from train.loss import AnnealedInfoNCE

    model.eval()
    loss_fn = AnnealedInfoNCE(tau_max=0.07, tau_min=0.07, anneal_steps=1)
    # Fixed tau at minimum for evaluation — no annealing during eval

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=collate_triplets,
    )

    total_loss        = 0
    total_align       = 0   # anchor · positive cosine sim
    total_discrim     = 0   # anchor · hard_neg cosine sim (want LOW)
    total_steps       = 0

    use_amp = torch.cuda.is_available() and str(device).startswith('cuda')
    autocast_ctx = torch.autocast('cuda') if use_amp else nullcontext()

    with torch.inference_mode():
        for batch in loader:
            def get_fingerprint(role):
                ids = batch[role]['token_ids'].to(device)
                P_e = batch[role]['P_electro'].to(device)
                P_s = batch[role]['P_steric'].to(device)
                with autocast_ctx:
                    projection = model(
                        ids,
                        P_electro=P_e,
                        P_steric=P_s,
                        return_projection=True,
                    )['projection']
                return F.normalize(projection, dim=-1)

            z_a = get_fingerprint('anchor')
            z_p = get_fingerprint('positive')
            z_hn = get_fingerprint('hard_neg')
            z_bg = get_fingerprint('bg_neg')

            _, metrics = loss_fn(z_a, z_p, z_hn, z_bg)

            # Cosine similarities
            align_sim  = (z_a * z_p).sum(dim=-1).mean().item()
            discrim_sim = (z_a * z_hn).sum(dim=-1).mean().item()

            total_loss    += metrics['loss']
            total_align   += align_sim
            total_discrim += discrim_sim
            total_steps   += 1

    results = {
        'loss':              total_loss    / max(total_steps, 1),
        'alignment':         total_align   / max(total_steps, 1),
        'discrimination':    total_discrim / max(total_steps, 1),
        'pharmacophore_gap': (total_align - total_discrim) / max(total_steps, 1),
    }

    print(f"\n=== {label} EVALUATION ===")
    print(f"  Loss              : {results['loss']:.4f}")
    print(f"  Alignment score   : {results['alignment']:.4f}  (anchor↔positive, want HIGH)")
    print(f"  Discrimination    : {results['discrimination']:.4f}  (anchor↔hard_neg, want LOW)")
    print(f"  Pharmacophore gap : {results['pharmacophore_gap']:.4f}  ← your KEY metric")

    if results['pharmacophore_gap'] < 0.1:
        print("  ⚠️  Gap too small — model not distinguishing pharmacophores yet")
    elif results['pharmacophore_gap'] > 0.4:
        print("  ✅ Strong pharmacophore discrimination")

    return results

def run_training(train_dataset, val_dataset, device='cpu',
                 tokenizer_path="./ncaa_tokenizer"):
    model_config = load_model_config(tokenizer_path)
    model_config['max_seq_len'] = 256

    model     = build_model(model_config).to(device)
    loss_fn   = AnnealedInfoNCE(
        tau_max=TRAINING_CONFIG['tau_max'],
        tau_min=TRAINING_CONFIG['tau_min'],
        anneal_steps=TRAINING_CONFIG['anneal_steps'],
    )
    optimizer = torch.optim.AdamW(
        get_parameter_groups(model),
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    # ── Learning rate scheduler — reduce on plateau ──────────────────
    # If val pharmacophore gap stops improving, halve the LR
    # This is more appropriate than a fixed schedule for small datasets
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',          # we want gap to go UP
        factor=0.5,          # halve LR on plateau
        patience=3,          # wait 3 evals before reducing
        min_lr=1e-7
    )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,        # Use multiple CPU cores to build batches simultaneously!
        pin_memory=True,
        collate_fn=collate_triplets,
    )

    best_gap      = -float('inf')
    best_epoch    = 0
    patience_count = 0
    eval_every    = TRAINING_CONFIG.get('eval_every', 2)
    patience      = TRAINING_CONFIG.get('early_stop_patience', 5)
    min_delta     = TRAINING_CONFIG.get('min_delta', 0.01)

    for epoch in range(TRAINING_CONFIG['epochs']):
        print(f"\n=== Epoch {epoch+1}/{TRAINING_CONFIG['epochs']} ===")
        train_metrics = train_epoch(model, loader, optimizer, loss_fn, device)
        print(f"Train avg loss: {train_metrics['avg_loss']:.4f}")

        # ── Early stopping trigger ────────────────────────────────────
        # Stop immediately if training loss goes negative
        # This is always overfit — do not continue
        if train_metrics['avg_loss'] < 0:
            print(f"\n🚨 NEGATIVE TRAINING LOSS DETECTED at epoch {epoch+1}")
            print(f"   This indicates complete memorization of training data.")
            print(f"   Stopping training. Load best model from ncaa_encoder_best.pt")
            break

        # ── Periodic evaluation ───────────────────────────────────────
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, val_dataset, device, label="VAL")
            gap = val_metrics['pharmacophore_gap']

            scheduler.step(gap)

            if gap > best_gap + min_delta:
                best_gap    = gap
                best_epoch  = epoch + 1
                patience_count = 0
                torch.save(model.state_dict(), "ncaa_encoder_best.pt")
                print(f"  💾 New best saved (gap={best_gap:.4f})")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{patience})")

                if patience_count >= patience:
                    print(f"\n⏹  Early stopping at epoch {epoch+1}")
                    print(f"   Best was epoch {best_epoch} with gap={best_gap:.4f}")
                    break

    print(f"\nTraining complete. Best epoch: {best_epoch}, gap={best_gap:.4f}")

    # Load best weights before returning
    import os
    if os.path.exists("ncaa_encoder_best.pt"):
        model.load_state_dict(torch.load("ncaa_encoder_best.pt"))
        print("Loaded best model weights")

    return model