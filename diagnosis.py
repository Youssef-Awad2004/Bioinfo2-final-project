# diagnosis.py
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from tokenizer.tokenizer import SmilesBPETokenizer
from model.encoder import NcAATransformerEncoder
from model.config import load_model_config

def diagnose_collapse(model_path: str = None, tokenizer_path: str = None):
    """
    Tests whether the model has collapsed to geometric poles
    or learned genuine chemical representations.
    
    A collapsed model will show:
      - All positives near cosine sim +1.0
      - All negatives near cosine sim -1.0
      - Near-zero variance within each group
      
    A genuine model will show:
      - Positives: 0.6-0.9 similarity (close but not perfect)
      - Negatives: 0.1-0.4 similarity (different but not opposite)
      - Meaningful variance — different molecule pairs score differently
    """
    # Resolve paths from env vars or defaults
    if model_path is None:
        PROJECT_DIR = Path(__file__).resolve().parent
        KAGGLE_WORKING_DIR = Path("/kaggle/working")
        DEFAULT_CACHE_DIR = KAGGLE_WORKING_DIR / "cache" if KAGGLE_WORKING_DIR.exists() else PROJECT_DIR / "cache"
        model_path = str(Path(os.getenv("BIOINFO_CACHE_DIR", str(DEFAULT_CACHE_DIR))) / "ncaa_encoder_best.pt")
    
    if tokenizer_path is None:
        PROJECT_DIR = Path(__file__).resolve().parent
        tokenizer_path = str(Path(os.getenv("BIOINFO_TOKENIZER_DIR", str(PROJECT_DIR / "ncaa_tokenizer"))))
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = SmilesBPETokenizer(pretrained_path=tokenizer_path)
    config    = load_model_config(tokenizer_path)
    config['max_seq_len'] = 256

    model = NcAATransformerEncoder(**config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def encode(smiles):
        ids = torch.tensor(
            [tokenizer.encode_smiles(smiles, max_length=256)]
        ).to(device)
        with torch.no_grad():
            proj = model(ids, return_projection=True)['projection']
        return F.normalize(proj, dim=-1).squeeze(0)

    # Pairs spanning a range of expected similarities
    test_pairs = [
        # Should be HIGH similarity — same molecule, different SMILES
        ("Identical mol A",
         "N[C@@H](Cc1ccc(F)cc1)C(=O)O",
         "OC(=O)[C@@H](N)Cc1ccc(F)cc1",
         "positive"),

        # Should be HIGH — same molecule
        ("Identical mol B",
         "N[C@@H](C[SeH])C(=O)O",
         "OC(=O)[C@@H](N)C[SeH]",
         "positive"),

        # Should be MODERATE-HIGH — close structural analogs
        ("F-Phe vs Cl-Phe",
         "N[C@@H](Cc1ccc(F)cc1)C(=O)O",
         "N[C@@H](Cc1ccc(Cl)cc1)C(=O)O",
         "analog"),

        # Should be MODERATE — same class, different side chain
        ("F-Phe vs Phe",
         "N[C@@H](Cc1ccc(F)cc1)C(=O)O",
         "N[C@@H](Cc1ccccc1)C(=O)O",
         "analog"),

        # Should be LOW-MODERATE — different amino acid class
        ("SeCys vs Phe",
         "N[C@@H](C[SeH])C(=O)O",
         "N[C@@H](Cc1ccccc1)C(=O)O",
         "different"),

        # Should be LOW — very different amino acids
        ("AzidoAla vs Phe",
         "N[C@@H](CN=[N+]=[N-])C(=O)O",
         "N[C@@H](Cc1ccccc1)C(=O)O",
         "different"),

        # Should be VERY LOW — amino acid vs non-amino acid
        ("Phe vs Aspirin",
         "N[C@@H](Cc1ccccc1)C(=O)O",
         "CC(=O)Oc1ccccc1C(=O)O",
         "unrelated"),

        # Should be VERY LOW — completely unrelated molecules
        ("Phe vs Caffeine",
         "N[C@@H](Cc1ccccc1)C(=O)O",
         "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
         "unrelated"),
    ]

    print("=== COLLAPSE DIAGNOSIS ===")
    print(f"{'Pair':35s} {'Expected':12s} {'Cosine Sim':>10s}  {'Status'}")
    print("-" * 75)

    sims_by_type = {'positive': [], 'analog': [], 'different': [], 'unrelated': []}
    any_collapsed = False

    for name, smi1, smi2, expected in test_pairs:
        z1 = encode(smi1)
        z2 = encode(smi2)
        sim = (z1 * z2).sum().item()
        sims_by_type[expected].append(sim)

        # Flag collapse indicators
        if expected == 'positive'  and sim < 0.7:
            status = "⚠️  LOW for same molecule"
        elif expected == 'unrelated' and sim > 0.5:
            status = "🚨 HIGH for unrelated — COLLAPSE"
            any_collapsed = True
        elif expected == 'unrelated' and sim < -0.5:
            status = "🚨 STRONGLY NEGATIVE — POLE COLLAPSE"
            any_collapsed = True
        elif expected == 'different' and sim < -0.3:
            status = "🚨 NEGATIVE — POLE COLLAPSE"
            any_collapsed = True
        else:
            status = "✅ reasonable"

        print(f"  {name:35s} {expected:12s} {sim:10.4f}  {status}")

    print("\n=== SIMILARITY DISTRIBUTION ===")
    import numpy as np
    for pair_type, sims in sims_by_type.items():
        if sims:
            print(f"  {pair_type:12s}: mean={np.mean(sims):.3f}  "
                  f"std={np.std(sims):.3f}  "
                  f"range=[{np.min(sims):.3f}, {np.max(sims):.3f}]")

    print("\n=== DIAGNOSIS ===")
    pos_mean = np.mean(sims_by_type['positive'])   if sims_by_type['positive']   else 0
    unr_mean = np.mean(sims_by_type['unrelated'])  if sims_by_type['unrelated']  else 0
    dif_mean = np.mean(sims_by_type['different'])  if sims_by_type['different']  else 0

    if unr_mean < -0.5 or dif_mean < -0.3:
        print("  🚨 POLE COLLAPSE CONFIRMED")
        print("     Model learned label lookup, not chemistry.")
        print("     Use epoch 5-10 checkpoint instead — see recommendations.")
        print("     Do NOT use ncaa_encoder_final.pt for screening.")
    elif pos_mean > 0.95 and unr_mean > 0.4:
        print("  🚨 UNIFORM COLLAPSE — everything looks similar")
        print("     Model outputs same vector regardless of input.")
    elif pos_mean > 0.7 and unr_mean < 0.3 and dif_mean < 0.5:
        print("  ✅ GENUINE LEARNING — model learned chemistry")
        print("     Safe to use for pharmacophore screening.")
    else:
        print("  ⚠️  AMBIGUOUS — partial learning, partial memorization")
        print("     Proceed with caution. Validate on external holdout.")


if __name__ == "__main__":
    diagnose_collapse()
