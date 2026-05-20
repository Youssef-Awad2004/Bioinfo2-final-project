# main.py - complete replacement

import os
from pathlib import Path
import torch
import pandas as pd
from dataPipeline.ingester import fetch_chembl_peptides, fetch_canonical_baselines
from dataPipeline.augmenter import run_augmentation_pipeline
from tokenizer.tokenizer import SmilesBPETokenizer
from train.dataset import MolecularTripletDataset
from train.train import run_training , evaluate
from model.config import load_model_config, TRAINING_CONFIG
from dataPipeline.validate import validate_pipeline_data
from dataPipeline.precompute_physics import ExactPhysicsLookup, build_exact_physics_cache



# ── Path configuration (Kaggle-compatible) ─────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
KAGGLE_WORKING_DIR = Path("/kaggle/working")

DEFAULT_CACHE_DIR = KAGGLE_WORKING_DIR / "cache" if KAGGLE_WORKING_DIR.exists() else PROJECT_DIR / "cache"
CACHE_DIR = Path(os.getenv("BIOINFO_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
TOKENIZER_DIR = Path(os.getenv("BIOINFO_TOKENIZER_DIR", str(PROJECT_DIR / "ncaa_tokenizer")))
OUTPUT_MODEL_PATH = Path(
    os.getenv(
        "BIOINFO_MODEL_OUT",
        str((KAGGLE_WORKING_DIR if KAGGLE_WORKING_DIR.exists() else PROJECT_DIR) / "ncaa_encoder_final.pt"),
    )
)

AUGMENTED_CACHE = CACHE_DIR / "augmented_targets.csv"
CANONICAL_CACHE = CACHE_DIR / "canonical_baselines.csv"
TRAIN_SPLIT = CACHE_DIR / "train.csv"
VAL_SPLIT = CACHE_DIR / "val.csv"
TEST_SPLIT = CACHE_DIR / "test.csv"
PHYSICS_CACHE = CACHE_DIR / "physics_cache_exact.pkl"
BEST_MODEL_PATH = CACHE_DIR / "ncaa_encoder_best.pt"


def build_or_load_data(force_rebuild: bool = False):
    """
    Builds the full dataset once and caches it to disk.
    Subsequent runs load from cache - no API calls needed.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if (not force_rebuild
            and TRAIN_SPLIT.exists()
            and VAL_SPLIT.exists()
            and TEST_SPLIT.exists()):
        print("Loading cached splits...")
        train_df = pd.read_csv(str(TRAIN_SPLIT))
        val_df = pd.read_csv(str(VAL_SPLIT))
        test_df = pd.read_csv(str(TEST_SPLIT))
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return train_df, val_df, test_df, all_data

    # ── Fetch ────────────────────────────────────────────────────────────
    print("[1/4] Ingesting ChEMBL peptides...")
    # Pull more - augmentation will expand this
    raw_target_df = fetch_chembl_peptides(limit=15000)

    print("[2/4] Augmenting targets...")
    augmented_df  = run_augmentation_pipeline(raw_target_df, pos_factor=5)

    print("[3/4] Ingesting UniProt canonicals...")
    # Match or exceed augmented target count for balanced in-batch negatives
    canonical_df  = fetch_canonical_baselines(limit=15000)

    augmented_df.to_csv(str(AUGMENTED_CACHE), index=False)
    canonical_df.to_csv(str(CANONICAL_CACHE), index=False)

    # ── Diagnostic check ─────────────────────────────────────────────────
    print("\n=== DATA AUDIT ===")
    print(augmented_df['type'].value_counts())
    n_anchors = (augmented_df['type'] == 'noncanonical_target').sum()
    if n_anchors == 0:
        raise ValueError(
            "FATAL: Zero noncanonical_target rows in augmented dataset.\n"
            "ChEMBL API returned no peptides. Check your internet connection\n"
            "and run: python -c \"from dataPipeline.ingester import fetch_chembl_peptides; print(fetch_chembl_peptides(limit=5))\""
        )

    # ── Split anchors into train/val/test BEFORE augmentation lookup ──────
    # Critical: split at the ANCHOR level, not the row level
    # If you split rows, the same molecule appears in train and test via positives

        # main.py - add after DATA AUDIT print
    print("\n[VALIDATION] Checking data quality...")
    val_report = validate_pipeline_data(
        augmented_df=augmented_df,
        canonical_df=canonical_df,
        verbose=True,       # True for full per-molecule detail
        )

    if not val_report['overall_pass']:
        print("\n[WARNING]  Data quality issues detected ")
        print("Training will continue but results may be unreliable")


    print("\n[4/4] Building train/val/test splits...")
    anchors = augmented_df[augmented_df['type'] == 'noncanonical_target'].copy()
    anchors = anchors.sample(frac=1, random_state=42).reset_index(drop=True)

    n       = len(anchors)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.15)
    # test gets the remainder

    train_ids = set(anchors.iloc[:n_train]['id'])
    val_ids   = set(anchors.iloc[n_train:n_train + n_val]['id'])
    test_ids  = set(anchors.iloc[n_train + n_val:]['id'])

    def get_split_df(anchor_ids: set, augmented_df, canonical_df) -> pd.DataFrame:
        """
        Returns all rows belonging to this split:
        - anchor rows with id in anchor_ids
        - positive/negative rows with anchor_id in anchor_ids
        - canonical baselines (shared across splits - they are background negatives)
        """
        mask_anchor = (
            (augmented_df['type'] == 'noncanonical_target') &
            (augmented_df['id'].isin(anchor_ids))
        )
        mask_aug = (
            (augmented_df['type'].isin(['positive_pair', 'hard_negative_pair'])) &
            (augmented_df['anchor_id'].isin(anchor_ids))
        )
        split_aug = augmented_df[mask_anchor | mask_aug].copy()
        return pd.concat([split_aug, canonical_df], ignore_index=True)

    train_df = get_split_df(train_ids, augmented_df, canonical_df)
    val_df   = get_split_df(val_ids,   augmented_df, canonical_df)
    test_df  = get_split_df(test_ids,  augmented_df, canonical_df)

    train_df.to_csv(str(TRAIN_SPLIT), index=False)
    val_df.to_csv(str(VAL_SPLIT),   index=False)
    test_df.to_csv(str(TEST_SPLIT),  index=False)

    print(f"\nSplit sizes (anchor molecules):")
    print(f"  Train : {len(train_ids)} anchors -> {len(train_df)} total rows")
    print(f"  Val   : {len(val_ids)} anchors -> {len(val_df)} total rows")
    print(f"  Test  : {len(test_ids)} anchors -> {len(test_df)} total rows")

    all_data = pd.concat([augmented_df, canonical_df], ignore_index=True)

    return train_df, val_df, test_df, all_data


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== NCAA SCREENING ENGINE - {device.upper()} ===\n")

    # ── Data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df, all_data = build_or_load_data(force_rebuild=True)

    # ── Tokenizer ────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = SmilesBPETokenizer(pretrained_path=str(TOKENIZER_DIR))

    # ── Physics cache ────────────────────────────────────────────────────
    print("\nPrecomputing physics matrices...")
    physics_index, tensor_dir = build_exact_physics_cache(
        df=all_data,
        tokenizer=tokenizer.tokenizer,
        max_length=256,
        smiles_col='smiles',
        cache_path=str(PHYSICS_CACHE),
        force_rebuild=False,
    )
    physics_lookup = ExactPhysicsLookup(physics_index, tensor_dir, max_length=256)

    # ── Datasets ─────────────────────────────────────────────────────────
    canonical_df = pd.read_csv(str(CANONICAL_CACHE))

    train_dataset = MolecularTripletDataset(
        augmented_df=train_df,
        canonical_df=canonical_df,
        tokenizer=tokenizer,
        physics_lookup=physics_lookup,
        max_length=256,      # ← increased from 128, see Fix 3
    )
    val_dataset = MolecularTripletDataset(
        augmented_df=val_df,
        canonical_df=canonical_df,
        tokenizer=tokenizer,
        physics_lookup=physics_lookup,
        max_length=256,
    )

    print(f"\nTrain triplets : {len(train_dataset)}")
    print(f"Val triplets   : {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty.\n"
            "Run with force_rebuild=True and check ChEMBL API response."
        )

    # ── Training ──────────────────────────────────────────────────────────
    model = run_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        tokenizer_path=str(TOKENIZER_DIR),
        best_model_path=str(BEST_MODEL_PATH),
    )

    # ── Final Test Evaluation ─────────────────────────────────────────────
    print("\n=== FINAL TEST EVALUATION ===")
    test_dataset = MolecularTripletDataset(
        augmented_df=test_df,
        canonical_df=canonical_df,
        tokenizer=tokenizer,
        max_length=256,
        physics_lookup=physics_lookup,
    )
    evaluate(model, test_dataset, device, label="TEST")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(OUTPUT_MODEL_PATH))
    print(f"\n[OK] Model saved to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()