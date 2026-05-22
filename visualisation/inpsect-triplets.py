"""
inspect_triplet.py
------------------
Analyse one anchor triplet from your validation CSV.

Usage:
    python inspect_triplet.py --csv val.csv --anchor ncAA_difluoro_Ala
    python inspect_triplet.py --csv val.csv  # picks a random named ncAA anchor
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize

# ── helpers ──────────────────────────────────────────────────────────────────

def get_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

def tanimoto(smi_a: str, smi_b: str) -> float | None:
    fa, fb = get_fp(smi_a), get_fp(smi_b)
    if fa is None or fb is None:
        return None
    return round(DataStructs.TanimotoSimilarity(fa, fb), 4)

def validate(smi: str) -> dict:
    """Basic chemical feasibility checks."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"valid_smiles": False}

    # Strip atom-map numbers (your hard negatives carry [:1] etc.)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    mw        = round(Descriptors.MolWt(mol), 2)
    hbd       = Descriptors.NumHDonors(mol)
    hba       = Descriptors.NumHAcceptors(mol)
    rotb      = Descriptors.NumRotatableBonds(mol)
    logp      = round(Descriptors.MolLogP(mol), 2)
    n_atoms   = mol.GetNumHeavyAtoms()

    # Amino-acid backbone: free amine + alpha carbon + carboxylic acid
    backbone_smarts = Chem.MolFromSmarts("[NX3;H2][CX4][CX3](=O)[OX2H1]")
    has_backbone    = mol.HasSubstructMatch(backbone_smarts) if backbone_smarts else None

    # Valence check (RDKit already enforces this on parse, but be explicit)
    try:
        Chem.SanitizeMol(mol)
        sane = True
    except Exception:
        sane = False

    # Canonical SMILES (removes stereo artefacts from augmentation)
    canonical = Chem.MolToSmiles(mol, canonical=True)

    return {
        "valid_smiles":   True,
        "sanitized":      sane,
        "canonical_smi":  canonical,
        "mw":             mw,
        "heavy_atoms":    n_atoms,
        "hbd":            hbd,
        "hba":            hba,
        "rotatable_bonds":rotb,
        "logP":           logp,
        "aa_backbone":    has_backbone,
    }

def same_molecule(smi_a: str, smi_b: str) -> bool:
    """True iff canonical SMILES match (ignoring map numbers / stereo writing)."""
    def canon(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    return canon(smi_a) == canon(smi_b)

def _mol_for_draw(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def draw_triplet(anchor_id: str, anchor_smi: str, positives: list[str],
                 negatives: list[str], out_path: str | None,
                 max_pos: int = 1, max_neg: int = 1) -> None:
    if max_pos < 0:
        max_pos = 0
    if max_neg < 0:
        max_neg = 0

    mols = []
    legends = []

    anchor_mol = _mol_for_draw(anchor_smi)
    if anchor_mol is None:
        print("[DRAW] Anchor SMILES is invalid; skipping image.")
        return

    mols.append(anchor_mol)
    legends.append(f"ANCHOR\n{anchor_smi}")

    count = 0
    for pos_smi in positives:
        if count >= max_pos:
            break
        pos_mol = _mol_for_draw(pos_smi)
        if pos_mol is None:
            continue
        count += 1
        mols.append(pos_mol)
        legends.append(f"POS {count}\n{pos_smi}")

    neg_count = 0
    for neg_smi in negatives:
        if neg_count >= max_neg:
            break
        neg_mol = _mol_for_draw(neg_smi)
        if neg_mol is None:
            continue
        neg_count += 1
        mols.append(neg_mol)
        legends.append(f"NEG {neg_count}\n{neg_smi}")

    if not mols:
        print("[DRAW] No molecules to draw.")
        return

    if out_path:
        out_file = Path(out_path)
    else:
        out_file = Path(__file__).resolve().parent / f"triplet_{anchor_id}.png"

    mols_per_row = min(len(mols), 3)
    if out_file.suffix.lower() == ".svg":
        svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            legends=legends,
            subImgSize=(350, 300),
            useSVG=True,
        )
        out_file.write_text(svg, encoding="utf-8")
    else:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            legends=legends,
            subImgSize=(350, 300),
        )
        img.save(str(out_file))

    print(f"[DRAW] Wrote image: {out_file}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="C:\\Users\\yousef\\Desktop\\College\\Bioinformatics_II\\Bioinfo2-final-project\\cache\\train.csv",        help="Path to val.csv")
    parser.add_argument("--anchor", default=None,             help="Anchor ID (e.g. ncAA_difluoro_Ala)")
    parser.add_argument("--draw",   action="store_true",      help="Draw anchor and positives to an image")
    parser.add_argument("--out",    default=None,             help="Output image path (.png or .svg)")
    parser.add_argument("--max_pos", type=int, default=1,      help="Max positive pairs to draw")
    parser.add_argument("--max_neg", type=int, default=1,      help="Max negative pairs to draw")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Pick anchor
    anchors = df[df["type"] == "noncanonical_target"]
    if args.anchor:
        row = anchors[anchors["id"] == args.anchor]
        if row.empty:
            sys.exit(f"Anchor '{args.anchor}' not found. Available: {anchors['id'].tolist()[:10]} …")
        anchor_id  = args.anchor
        anchor_smi = row.iloc[0]["smiles"]
    else:
        row        = anchors[anchors["id"].str.startswith("ncAA_")].sample(1)
        anchor_id  = row.iloc[0]["id"]
        anchor_smi = row.iloc[0]["smiles"]

    positives  = df[(df["anchor_id"] == anchor_id) & (df["type"] == "positive_pair")]["smiles"].tolist()
    hard_negs  = df[(df["anchor_id"] == anchor_id) & (df["type"] == "hard_negative_pair")]["smiles"].tolist()

    print("=" * 60)
    print(f"ANCHOR  : {anchor_id}")
    print(f"SMILES  : {anchor_smi}")
    print(f"Positives found : {len(positives)}")
    print(f"Hard negatives  : {len(hard_negs)}")
    print("=" * 60)

    # ── 1. Anchor validity ────────────────────────────────────────
    print("\n── Anchor validation ──")
    v = validate(anchor_smi)
    for k, val in v.items():
        print(f"  {k:<22}: {val}")

    # ── 2. Positive pairs ─────────────────────────────────────────
    print("\n── Positive pairs (should all be the same molecule) ──")
    for i, pos_smi in enumerate(positives):
        sim   = tanimoto(anchor_smi, pos_smi)
        same  = same_molecule(anchor_smi, pos_smi)
        valid = validate(pos_smi)["valid_smiles"]
        print(f"  [{i+1}] Tanimoto={sim}  same_mol={same}  valid={valid}  {pos_smi}")

    # ── 3. Hard negatives ─────────────────────────────────────────
    print("\n── Hard negatives (similar but not identical) ──")
    sims = []
    for i, neg_smi in enumerate(hard_negs):
        sim  = tanimoto(anchor_smi, neg_smi)
        same = same_molecule(anchor_smi, neg_smi)
        v2   = validate(neg_smi)
        sims.append(sim)
        bb   = v2.get("aa_backbone", "?")
        print(f"  [{i+1}] Tanimoto={sim}  same_mol={same}  backbone={bb}  {neg_smi}")

    if sims:
        valid_sims = [s for s in sims if s is not None]
        print(f"\n  Similarity stats  mean={round(sum(valid_sims)/len(valid_sims),4)}"
              f"  min={min(valid_sims)}  max={max(valid_sims)}")

    # ── 4. Quick sanity summary ───────────────────────────────────
    pos_tanimotos = [tanimoto(anchor_smi, s) for s in positives]
    all_pos_same  = all(same_molecule(anchor_smi, s) for s in positives)
    neg_tanimotos = [s for s in sims if s is not None]

    print("\n── Sanity summary ──")
    print(f"  All positives canonically identical : {all_pos_same}  ✓" if all_pos_same else
          f"  All positives canonically identical : {all_pos_same}  ✗ CHECK AUGMENTATION")
    if neg_tanimotos:
        avg_neg = sum(neg_tanimotos) / len(neg_tanimotos)
        hard_enough = all(s < 0.9 for s in neg_tanimotos)
        print(f"  Avg anchor↔neg Tanimoto            : {round(avg_neg,4)}")
        print(f"  All negatives below 0.9 threshold  : {hard_enough}  " +
              ("✓" if hard_enough else "✗ Some negatives may be too similar"))
    print("=" * 60)

    if args.draw:
        draw_triplet(
            anchor_id,
            anchor_smi,
            positives,
            hard_negs,
            args.out,
            args.max_pos,
            args.max_neg,
        )

if __name__ == "__main__":
    main()