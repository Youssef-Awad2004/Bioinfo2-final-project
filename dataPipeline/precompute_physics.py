# dataPipeline/precompute_physics.py
#
# Precomputes P_electro and P_steric matrices for every unique molecule
# in your dataset. Stores as a lookup table keyed by canonical SMILES.
# Run once before training. Training loop does a dictionary lookup
# instead of recomputing RDKit properties every forward pass.

import torch
import numpy as np
import pandas as pd
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional


_C_EN    = 2.55   # Pauling EN of Carbon
_F_EN    = 3.98   # Pauling EN of Fluorine (most electronegative)
_MAX_GAP = _F_EN - _C_EN   # 1.43 — normalization denominator

ELECTRONEGATIVITY_FALLBACK = {
     1: -0.2448,   # H   Pauling EN = 2.20
     3: -1.0979,   # Li  Pauling EN = 0.98
     4: -0.6853,   # Be  Pauling EN = 1.57
     5: -0.3566,   # B   Pauling EN = 2.04  (borono-ncAAs)
     6:  0.0000,   # C   Pauling EN = 2.55  (reference)
     7: +0.3427,   # N   Pauling EN = 3.04
     8: +0.6224,   # O   Pauling EN = 3.44
     9: +1.0000,   # F   Pauling EN = 3.98  (fluorinated ncAAs)
    11: -1.1329,   # Na  Pauling EN = 0.93
    12: -0.8671,   # Mg  Pauling EN = 1.31
    13: -0.6573,   # Al  Pauling EN = 1.61
    14: -0.4545,   # Si  Pauling EN = 1.90
    15: -0.2517,   # P   Pauling EN = 2.19  (phosphono-ncAAs)
    16: +0.0210,   # S   Pauling EN = 2.58
    17: +0.4266,   # Cl  Pauling EN = 3.16
    19: -1.2098,   # K   Pauling EN = 0.82
    20: -1.0839,   # Ca  Pauling EN = 1.00
    31: -0.5175,   # Ga  Pauling EN = 1.81
    33: -0.2587,   # As  Pauling EN = 2.18
    34:  0.0000,   # Se  Pauling EN = 2.55  (selenocysteine — same as C)
    35: +0.2867,   # Br  Pauling EN = 2.96
    53: +0.0769,   # I   Pauling EN = 2.66
}

# Van der Waals radii — Bondi (1964), J. Phys. Chem.
VDW_RADII = {
    1:  1.20,   # H
    6:  1.70,   # C
    7:  1.55,   # N
    8:  1.52,   # O
    9:  1.47,   # F
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    34: 1.90,   # Se
    35: 1.85,   # Br
    53: 1.98,   # I
}
DEFAULT_VDW = 1.70



def compute_normalized_matrices(
    mol
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes normalized P_electro and P_steric at atom resolution.

    Charge strategy (in priority order):
      1. Gasteiger charge from RDKit — used when computation succeeds
      2. Pauling electronegativity fallback — used when Gasteiger returns
         NaN for ANY atom in the molecule (e.g. selenocysteine where
         the entire molecule fails, not just the Se atom)
      3. 0.0 — used for elements not in either table

    Steric strategy:
      Always uses Bondi vdW radii — no RDKit computation needed,
      no failure modes, purely from the hardcoded table.

    Normalization:
      Both matrices normalized to [-1, +1] at precompute time.
      No normalization needed at training time.
    """

    # ── Step 1: Attempt Gasteiger charges ───────────────────────────────
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gasteiger_ok = True
    except Exception:
        gasteiger_ok = False

    # ── Step 2: Collect charges with fallback logic ─────────────────────
    charges = []
    radii   = []
    used_fallback = []

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()

        # Attempt Gasteiger
        charge = None
        if gasteiger_ok:
            try:
                q = atom.GetDoubleProp('_GasteigerCharge')
                if q == q and abs(q) > 1e-10:
                    # Valid non-NaN non-zero Gasteiger charge
                    charge = q
            except Exception:
                pass

        # Fallback to Pauling electronegativity
        if charge is None:
            charge = ELECTRONEGATIVITY_FALLBACK.get(atomic_num, 0.0)
            used_fallback.append(atom.GetSymbol())

        charges.append(charge)
        radii.append(VDW_RADII.get(atomic_num, DEFAULT_VDW))

    # ── Step 3: Build pairwise matrices ─────────────────────────────────
    charges_t = torch.tensor(charges, dtype=torch.float32)
    radii_t   = torch.tensor(radii,   dtype=torch.float32)

    P_electro = charges_t.unsqueeze(1) - charges_t.unsqueeze(0)
    P_steric  = radii_t.unsqueeze(1)   + radii_t.unsqueeze(0)

    # ── Step 4: Normalize to [-1, +1] ───────────────────────────────────
    def safe_normalize(matrix: torch.Tensor) -> torch.Tensor:
        abs_max = matrix.abs().max()
        return matrix / abs_max if abs_max > 1e-6 else matrix

    return safe_normalize(P_electro), safe_normalize(P_steric)


def build_physics_lookup(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    cache_path: str = './cache/physics_lookup.pkl',
    force_rebuild: bool = False,
) -> dict:
    """
    Builds a lookup table: canonical_smiles → (P_electro, P_steric)
    
    Both matrices are at atom resolution [N_atoms, N_atoms].
    The token-space projection happens at training time using the
    current SMILES variant's token alignment — this is the key design decision
    that makes precomputation correct across SMILES variants.
    
    Returns dict with structure:
        {
            canonical_smiles: {
                'P_electro': torch.Tensor [N_atoms, N_atoms],
                'P_steric':  torch.Tensor [N_atoms, N_atoms],
                'n_atoms':   int,
            }
        }
    """
    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading physics lookup from {cache_path}...")
        with open(cache_path, 'rb') as f:
            lookup = pickle.load(f)
        print(f"Loaded {len(lookup)} precomputed molecules")
        return lookup

    print(f"Precomputing physics matrices for {len(df)} rows...")

    lookup   = {}
    failed   = []
    seen     = set()

    smiles_list = df[smiles_col].dropna().tolist()
    total = len(smiles_list)

    for i, smiles in enumerate(smiles_list):
        if i % 1000 == 0:
            print(f"  {i}/{total} ({100*i/total:.1f}%)")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(smiles)
            continue

        # Key by canonical SMILES — this is the critical design choice
        # All SMILES variants of the same molecule map to one canonical key
        canonical = Chem.MolToSmiles(mol, canonical=True)

        # Skip if already computed for this molecule
        if canonical in seen:
            continue
        seen.add(canonical)

        # Use the canonical mol for consistent atom ordering
        canonical_mol = Chem.MolFromSmiles(canonical)
        if canonical_mol is None:
            failed.append(smiles)
            continue

        result = compute_normalized_matrices(canonical_mol)
        if result is None:
            failed.append(smiles)
            continue

        P_electro, P_steric = result
        lookup[canonical] = {
            'P_electro': P_electro,   # [N_atoms, N_atoms]
            'P_steric':  P_steric,    # [N_atoms, N_atoms]
            'n_atoms':   canonical_mol.GetNumAtoms(),
        }

    print(f"\nPrecomputation complete:")
    print(f"  Unique molecules computed : {len(lookup)}")
    print(f"  Failed                    : {len(failed)}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(lookup, f)
    print(f"  Saved to {cache_path}")

    return lookup