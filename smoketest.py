# smoke_test.py - full corrected version

import os
from pathlib import Path
import torch
from tokenizer.tokenizer import SmilesBPETokenizer
from model.encoder import NcAATransformerEncoder
from model.weight_transfer import transfer_chemberta_weights
from train.loss import AnnealedInfoNCE
from model.config import load_model_config, TRAINING_CONFIG


def run_smoke_test(tokenizer_path: str = None):
    if tokenizer_path is None:
        PROJECT_DIR = Path(__file__).resolve().parent
        tokenizer_path = str(Path(os.getenv("BIOINFO_TOKENIZER_DIR", str(PROJECT_DIR / "ncaa_tokenizer"))))
    
    print("=== SMOKE TEST ===\n")

    # ── Device check ────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("  [WARNING] CUDA not available - running on CPU")
        print("  Run: nvidia-smi  to check your CUDA version")
        print("  Then reinstall PyTorch with the correct CUDA build")
        print("  Continuing smoke test on CPU - functionally identical\n")

    # ── 1. Tokenizer ────────────────────────────────────────────────────
    print("[1/5] Loading tokenizer...")
    tokenizer  = SmilesBPETokenizer(pretrained_path=tokenizer_path)
    
    # -- 2. Config - read from tokenizer, not hardcoded ----
    print("[2/5] Loading model config from tokenizer...")
    MODEL_CONFIG = load_model_config(tokenizer_path)
    
    print(f"     vocab_size   : {MODEL_CONFIG['vocab_size']}")
    print(f"     pad_token_id : {MODEL_CONFIG['pad_token_id']}")
    print(f"     embed_dim    : {MODEL_CONFIG['embed_dim']}")

    # Sanity check - tokenizer and config must agree
    assert tokenizer.vocab_size == MODEL_CONFIG['vocab_size'], \
        f"Vocab mismatch: {tokenizer.vocab_size} vs {MODEL_CONFIG['vocab_size']}"
    print("     [OK] Tokenizer and config vocab sizes match")

    # ── 3. Model + Weight Transfer ──────────────────────────────────────
    print("\n[3/5] Building model and transferring ChemBERTa weights...")
    model = NcAATransformerEncoder(**MODEL_CONFIG).to(device)
    model = transfer_chemberta_weights(model)
    
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Total parameters    : {total_params:,}")
    print(f"     Trainable           : {trainable_params:,}")

    # ── 4. Fake Batch ───────────────────────────────────────────────────
    print("\n[4/5] Building fake batch...")
    B = 4
    T = MODEL_CONFIG['max_seq_len']

    fake_ids = torch.randint(0, MODEL_CONFIG['vocab_size'], (B * 4, T)).to(device)
    fake_P_e = torch.randn(B * 4, T, T).to(device)
    fake_P_s = torch.abs(torch.randn(B * 4, T, T)).to(device)

    # ── 5. Forward Pass ─────────────────────────────────────────────────
    print("\n[5/5] Running forward pass...")
    model.train()

    try:
        output = model(
            fake_ids,
            P_electro=fake_P_e,
            P_steric=fake_P_s,
            return_projection=True
        )
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        raise

    fingerprint = output['fingerprint']
    projection  = output['projection']

    print(f"     Input shape       : {fake_ids.shape}")
    print(f"     Fingerprint shape : {fingerprint.shape}")
    print(f"     Projection shape  : {projection.shape}")

    assert fingerprint.shape == (B * 4, MODEL_CONFIG['embed_dim']), \
        f"Fingerprint shape wrong: {fingerprint.shape}"
    assert projection.shape  == (B * 4, MODEL_CONFIG['projection_dim']), \
        f"Projection shape wrong: {projection.shape}"
    print("     [OK] Output shapes correct")

    # ── 6. Loss + Backward ──────────────────────────────────────────────
    print("\n[6/6] Running loss backward pass...")
    loss_fn = AnnealedInfoNCE(
        tau_max=TRAINING_CONFIG['tau_max'],
        tau_min=TRAINING_CONFIG['tau_min'],
        anneal_steps=TRAINING_CONFIG['anneal_steps']
    )

    z_a, z_p, z_hn, z_bg = projection.chunk(4, dim=0)
    loss, metrics = loss_fn(z_a, z_p, z_hn, z_bg)
    loss.backward()

    grad_check = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"     Loss       : {metrics['loss']:.4f}")
    print(f"     Tau        : {metrics['tau']:.4f}")
    print(f"     Gradients  : {'[OK] flowing' if grad_check else '[FAIL] MISSING'}")

    assert not torch.isnan(loss), "Loss is NaN - check P_electro computation"
    assert grad_check, "Gradients not flowing — check for detached tensors"

    # ── Real SMILES test ─────────────────────────────────────────────────
    print("\n[BONUS] Real SMILES forward pass...")
    test_smiles = [
        ("Selenocysteine",  "C([SeH])C(N)C(=O)O"),
        ("Azidoalanine",    "N=[N+]=[N-]CC(N)C(=O)O"),
        ("Beta-alanine",    "NCCC(=O)O"),
        ("Alanine",         "CC(N)C(=O)O"),
    ]

    model.eval()
    with torch.no_grad():
        for name, smi in test_smiles:
            ids = torch.tensor(
                [tokenizer.encode_smiles(smi, max_length=T)]
            ).to(device)
            out  = model(ids, return_projection=False)
            norm = out['fingerprint'].norm().item()
            print(f"     {name:20s} → fingerprint norm: {norm:.4f}")

    print("\n" + "="*50)
    print("✅ SMOKE TEST PASSED")
    print("="*50)
    print(f"\nCopy these into trainer.py:")
    print(f"  VOCAB_SIZE   = {MODEL_CONFIG['vocab_size']}")
    print(f"  PAD_TOKEN_ID = {MODEL_CONFIG['pad_token_id']}")


if __name__ == "__main__":
    run_smoke_test()