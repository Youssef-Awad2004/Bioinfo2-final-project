# dataPipeline/validate_data.py
#
# Scientific data validation for the ncAA screening pipeline.
#
# Answers two questions with hard chemical evidence:
#   1. Is my ncAA data actually amino acids and not random molecules?
#   2. Are my hard negatives physically possible decoys or chemical garbage?
#
# Run standalone:  python validate_data.py
# Run from main:   from dataPipeline.validate_data import validate_pipeline_data

from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, rdFingerprintGenerator
import pandas as pd
from collections import defaultdict


# ── SMARTS Patterns ──────────────────────────────────────────────────────────
# Two backbone patterns to cover both alpha and beta amino acids

_ALPHA_AA = Chem.MolFromSmarts(
    '[NX3;H2,H1,H0][CX4][CX3](=O)[OX2H1,OX1-,OX2]'
)
_BETA_AA = Chem.MolFromSmarts(
    '[NX3;H2,H1,H0][CX4][CX4][CX3](=O)[OX2H1,OX1-,OX2]'
)
_CARBOXYL = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
_AMINE    = Chem.MolFromSmarts('[NX3;H2,H1]')

# Tanimoto thresholds - calibrated against known amino acid pairs
# FPhe vs Ala = 0.276, SeCys vs Ala = 0.400, Ala vs Gly = 0.294
# Floor set at 0.20: below this the decoy shares nothing with the anchor
# Ceiling set at 0.95: above this the mutation did nothing meaningful
TANIMOTO_FLOOR   = 0.20
TANIMOTO_CEILING = 0.95
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Molecular weight range for amino acids and peptide-sized ncAA entries
# Gly = 75 Da (smallest), current dataset includes peptide-sized entries up to ~2.5 kDa
MW_MIN = 60.0
MW_MAX = 3000.0


# ── Core Validators ──────────────────────────────────────────────────────────

def validate_ncaa(smiles: str, source: str | None = None) -> tuple[bool, str]:
    """
    Checks whether a SMILES string represents a valid amino acid.

    Validation criteria:
      1. RDKit can parse and sanitize the SMILES
      2. Molecular weight is in the amino acid range [60, 800] Da
      3. Contains an amino acid backbone (alpha or beta)
         If backbone match fails, falls back to requiring both
         a free amine AND a carboxylic acid - catches unusual ncAA scaffolds

    Returns (is_valid: bool, reason: str)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, 'INVALID_SMILES'

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, f'SANITIZATION_FAILED:{e}'

    mw = Descriptors.MolWt(mol)
    # Allow source-specific molecular weight ceilings.
    # Default uses global MW_MIN / MW_MAX, but curated seeds may be smaller
    # while canonical baselines and ChEMBL peptides can be peptide-sized.
    max_mw = MW_MAX
    if source:
        s = source.lower()
        if 'seed' in s or 'curated' in s:
            max_mw = 1500.0
        elif 'canonical' in s or 'uniprot' in s:
            max_mw = 3000.0
        elif 'chembl' in s or 'noncanonical' in s or 'peptide' in s:
            max_mw = 3000.0

    if mw < MW_MIN or mw > max_mw:
        return False, f'MW_OUT_OF_RANGE:{mw:.0f}Da'

    has_alpha    = mol.HasSubstructMatch(_ALPHA_AA)
    has_beta     = mol.HasSubstructMatch(_BETA_AA)
    has_carboxyl = mol.HasSubstructMatch(_CARBOXYL)
    has_amine    = mol.HasSubstructMatch(_AMINE)

    if has_alpha or has_beta:
        return True, 'VALID_AA_BACKBONE'

    # Fallback: both amine and carboxylate present
    # Catches unusual ring systems like pipecolic acid where
    # the backbone SMARTS misses due to ring closure
    if has_carboxyl and has_amine:
        return True, 'VALID_AMINE_CARBOXYL'

    return False, 'NO_AA_BACKBONE'


def validate_hard_negative(
    anchor_smiles: str,
    neg_smiles: str
) -> tuple[bool, str]:
    """
    Checks whether a hard negative is a valid pharmacophore decoy.

    Validation criteria:
      1. RDKit can parse and sanitize the negative SMILES
      2. Is not identical to the anchor (mutation must have done something)
      3. Tanimoto similarity to anchor is in [0.20, 0.95]:
         - Below 0.20: too dissimilar - not a decoy, just noise
         - Above 0.95: too similar - mutation was cosmetic, not pharmacophoric
      4. Molecular weight ratio anchor/negative < 3.0
         (prevents cases where the negative lost half the molecule)

    Returns (is_valid: bool, reason: str)
    """
    anchor_mol = Chem.MolFromSmiles(anchor_smiles)
    neg_mol    = Chem.MolFromSmiles(neg_smiles)

    if neg_mol is None:
        return False, 'INVALID_SMILES'

    try:
        Chem.SanitizeMol(neg_mol)
    except Exception as e:
        return False, f'SANITIZATION_FAILED:{e}'

    # Identity check
    if Chem.MolToSmiles(anchor_mol) == Chem.MolToSmiles(neg_mol):
        return False, 'IDENTICAL_TO_ANCHOR'

    # Tanimoto similarity - Morgan fingerprints radius 2
    fp_a = _MORGAN_GENERATOR.GetFingerprint(anchor_mol)
    fp_n = _MORGAN_GENERATOR.GetFingerprint(neg_mol)
    tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_n)

    if tanimoto < TANIMOTO_FLOOR:
        return False, f'TOO_DISSIMILAR:tanimoto={tanimoto:.3f}'
    if tanimoto > TANIMOTO_CEILING:
        return False, f'COSMETIC_MUTATION:tanimoto={tanimoto:.3f}'

    # MW ratio check
    mw_a = Descriptors.MolWt(anchor_mol)
    mw_n = Descriptors.MolWt(neg_mol)
    ratio = max(mw_a, mw_n) / max(min(mw_a, mw_n), 1.0)
    if ratio > 3.0:
        return False, f'MW_RATIO_TOO_HIGH:{ratio:.1f}'

    return True, f'VALID:tanimoto={tanimoto:.3f}'


def validate_positive(
    anchor_smiles: str,
    pos_smiles: str
) -> tuple[bool, str]:
    """
    Checks whether a positive pair (SMILES enumeration variant) is valid.

    A positive must:
      1. Be parseable by RDKit
      2. Be the same molecule as the anchor (same canonical SMILES)
         SMILES enumeration should produce synonymous strings, not new molecules
      3. Not be literally identical as a string (that would mean
         doRandom=True produced no variation - degenerate augmentation)

    Returns (is_valid: bool, reason: str)
    """
    anchor_mol = Chem.MolFromSmiles(anchor_smiles)
    pos_mol    = Chem.MolFromSmiles(pos_smiles)

    if pos_mol is None:
        return False, 'INVALID_SMILES'

    # Canonical SMILES must match - same molecule
    canonical_anchor = Chem.MolToSmiles(anchor_mol)
    canonical_pos    = Chem.MolToSmiles(pos_mol)

    if canonical_anchor != canonical_pos:
        return False, 'DIFFERENT_MOLECULE'

    # String must differ - otherwise augmentation did nothing
    if anchor_smiles == pos_smiles:
        return False, 'IDENTICAL_STRING'

    return True, 'VALID'


# ── Full Pipeline Validator ───────────────────────────────────────────────────

def validate_pipeline_data(
    augmented_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    verbose: bool = True,
    sample_size: int = None,
) -> dict:
    """
    Runs all three validators across your full pipeline dataframes.

    Args:
        augmented_df:  Output of run_augmentation_pipeline()
                       Must contain columns: id, smiles, type, anchor_id
        canonical_df:  Output of fetch_canonical_baselines()
        verbose:       Print per-molecule failures
        sample_size:   If set, validate a random sample of each type
                       (useful for large datasets - None = validate all)

    Returns dict with keys:
        ncaa_pass_rate, ncaa_failures
        positive_pass_rate, positive_failures
        negative_pass_rate, negative_failures
        canonical_pass_rate, canonical_failures
        overall_pass: bool
        recommendations: list of strings
    """
    results = defaultdict(list)
    recommendations = []

    # ── 1. Validate ncAA Anchors ─────────────────────────────────────────
    anchors = augmented_df[augmented_df['type'] == 'noncanonical_target'].copy()
    if sample_size:
        anchors = anchors.sample(min(sample_size, len(anchors)), random_state=42)

    anchor_pass = 0
    anchor_fail_reasons = defaultdict(int)

    for _, row in anchors.iterrows():
        # Anchors originate from augmented_df and represent noncanonical targets
        valid, reason = validate_ncaa(row['smiles'], source='noncanonical_target')
        if valid:
            anchor_pass += 1
        else:
            anchor_fail_reasons[reason] += 1
            results['ncaa_failures'].append({
                'id': row['id'], 'smiles': row['smiles'], 'reason': reason
            })
            if verbose:
                print(f"  [ncAA FAIL] {row['id']}: {reason}")

    ncaa_rate = anchor_pass / max(len(anchors), 1)

    # ── 2. Validate Positive Pairs ───────────────────────────────────────
    positives = augmented_df[augmented_df['type'] == 'positive_pair'].copy()
    if sample_size:
        positives = positives.sample(min(sample_size, len(positives)), random_state=42)

    pos_pass = 0
    pos_fail_reasons = defaultdict(int)

    # Build anchor lookup
    anchor_smiles_map = dict(
        zip(anchors['id'], anchors['smiles'])
    )
    # Also include all anchors (not just sampled)
    all_anchors = augmented_df[augmented_df['type'] == 'noncanonical_target']
    anchor_smiles_map.update(dict(zip(all_anchors['id'], all_anchors['smiles'])))

    for _, row in positives.iterrows():
        anchor_smi = anchor_smiles_map.get(row.get('anchor_id', ''))
        if not anchor_smi:
            pos_fail_reasons['NO_ANCHOR_FOUND'] += 1
            results['positive_failures'].append({
                'id': row['id'], 'smiles': row['smiles'],
                'reason': 'NO_ANCHOR_FOUND'
            })
            continue

        valid, reason = validate_positive(anchor_smi, row['smiles'])
        if valid:
            pos_pass += 1
        else:
            pos_fail_reasons[reason] += 1
            results['positive_failures'].append({
                'id': row['id'], 'smiles': row['smiles'], 'reason': reason
            })
            if verbose:
                print(f"  [POS FAIL] {row['id']}: {reason}")

    pos_rate = pos_pass / max(len(positives), 1)

    # ── 3. Validate Hard Negatives ───────────────────────────────────────
    negatives = augmented_df[augmented_df['type'] == 'hard_negative_pair'].copy()
    if sample_size:
        negatives = negatives.sample(min(sample_size, len(negatives)), random_state=42)

    neg_pass = 0
    neg_fail_reasons = defaultdict(int)
    tanimoto_values  = []

    for _, row in negatives.iterrows():
        anchor_smi = anchor_smiles_map.get(row.get('anchor_id', ''))
        if not anchor_smi:
            neg_fail_reasons['NO_ANCHOR_FOUND'] += 1
            continue

        valid, reason = validate_hard_negative(anchor_smi, row['smiles'])
        if valid:
            neg_pass += 1
            # Extract tanimoto from reason string for statistics
            try:
                t = float(reason.split('tanimoto=')[1])
                tanimoto_values.append(t)
            except Exception:
                pass
        else:
            neg_fail_reasons[reason.split(':')[0]] += 1
            results['negative_failures'].append({
                'id': row['id'], 'smiles': row['smiles'],
                'anchor': anchor_smi, 'reason': reason
            })
            if verbose:
                print(f"  [NEG FAIL] {row['id']}: {reason}")

    neg_rate = neg_pass / max(len(negatives), 1)

    # ── 4. Validate Canonical Baselines ─────────────────────────────────
    cans = canonical_df.copy()
    if sample_size:
        cans = cans.sample(min(sample_size, len(cans)), random_state=42)

    can_pass = 0
    can_fail_reasons = defaultdict(int)

    for _, row in cans.iterrows():
        # Canonical baselines come from UniProt and can be peptide-length
        valid, reason = validate_ncaa(row['smiles'], source='canonical_baseline')
        if valid:
            can_pass += 1
        else:
            can_fail_reasons[reason] += 1
            results['canonical_failures'].append({
                'id': row['id'], 'smiles': row['smiles'], 'reason': reason
            })

    can_rate = can_pass / max(len(cans), 1)

    # ── Build Report ─────────────────────────────────────────────────────
    import numpy as np

    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)

    print(f"\n[1] ncAA Anchors ({len(anchors)} checked)")
    print(f"    Pass rate : {ncaa_rate*100:.1f}%  ({anchor_pass}/{len(anchors)})")
    if anchor_fail_reasons:
        print(f"    Failures  : {dict(anchor_fail_reasons)}")
    if ncaa_rate < 0.85:
        recommendations.append(
            "LOW ncAA pass rate - your ChEMBL/PubChem source is "
            "returning non-amino-acid molecules. Review ingester filters."
        )

    print(f"\n[2] Positive Pairs ({len(positives)} checked)")
    print(f"    Pass rate : {pos_rate*100:.1f}%  ({pos_pass}/{len(positives)})")
    if pos_fail_reasons:
        print(f"    Failures  : {dict(pos_fail_reasons)}")
    if pos_rate < 0.95:
        recommendations.append(
            "Positive pairs failing - SMILES enumeration is producing "
            "chemically different molecules. Check RDKit doRandom=True usage."
        )

    print(f"\n[3] Hard Negatives ({len(negatives)} checked)")
    print(f"    Pass rate : {neg_rate*100:.1f}%  ({neg_pass}/{len(negatives)})")
    if neg_fail_reasons:
        print(f"    Failures  : {dict(neg_fail_reasons)}")
    if tanimoto_values:
        print(f"    Tanimoto  : mean={np.mean(tanimoto_values):.3f}  "
              f"min={np.min(tanimoto_values):.3f}  "
              f"max={np.max(tanimoto_values):.3f}")
        if np.mean(tanimoto_values) < 0.3:
            recommendations.append(
                "Hard negatives have low mean Tanimoto (<0.30) - "
                "your SMARTS mutations are producing molecules too dissimilar "
                "to be meaningful decoys. The contrastive task is too easy."
            )
        if np.mean(tanimoto_values) > 0.85:
            recommendations.append(
                "Hard negatives have high mean Tanimoto (>0.85) - "
                "mutations are too conservative. Model will not learn "
                "fine pharmacophore differences."
            )
    if neg_rate < 0.70:
        recommendations.append(
            "Many hard negatives failing - SMARTS mutations are creating "
            "physically impossible molecules. Review augmenter.py mutation rules."
        )

    print(f"\n[4] Canonical Baselines ({len(cans)} checked)")
    print(f"    Pass rate : {can_rate*100:.1f}%  ({can_pass}/{len(cans)})")
    if can_fail_reasons:
        print(f"    Failures  : {dict(can_fail_reasons)}")

    overall = (
        ncaa_rate >= 0.85 and
        pos_rate  >= 0.95 and
        neg_rate  >= 0.70 and
        can_rate  >= 0.85
    )

    print(f"\n{'='*60}")
    print(f"OVERALL: {'[OK] PASS - data is scientifically valid' if overall else '[FAIL] FAIL - review recommendations below'}")
    if recommendations:
        print(f"\nRECOMMENDATIONS:")
        for i, r in enumerate(recommendations, 1):
            print(f"  {i}. {r}")
    print("="*60)

    return {
        'ncaa_pass_rate':      ncaa_rate,
        'positive_pass_rate':  pos_rate,
        'negative_pass_rate':  neg_rate,
        'canonical_pass_rate': can_rate,
        'ncaa_failures':       results['ncaa_failures'],
        'positive_failures':   results['positive_failures'],
        'negative_failures':   results['negative_failures'],
        'canonical_failures':  results['canonical_failures'],
        'tanimoto_values':     tanimoto_values,
        'overall_pass':        overall,
        'recommendations':     recommendations,
    }


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    print("Loading cached data...")
    try:
        augmented_df = pd.read_csv("./cache/augmented_targets.csv")
        canonical_df = pd.read_csv("./cache/canonical_baselines.csv")
        print(f"  augmented_targets : {len(augmented_df)} rows")
        print(f"  canonical_baselines: {len(canonical_df)} rows")
        print(f"  type distribution:")
        print(augmented_df['type'].value_counts().to_string())
    except FileNotFoundError:
        print("Cache not found - run main.py first to generate data")
        sys.exit(1)

    validate_pipeline_data(
        augmented_df=augmented_df,
        canonical_df=canonical_df,
        verbose=True,
        sample_size=None,   # validate everything
    )