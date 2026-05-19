import os
from pathlib import Path
import pandas as pd

# Kaggle-compatible paths
PROJECT_DIR = Path(__file__).resolve().parent
KAGGLE_WORKING_DIR = Path("/kaggle/working")
DEFAULT_CACHE_DIR = KAGGLE_WORKING_DIR / "cache" if KAGGLE_WORKING_DIR.exists() else PROJECT_DIR / "cache"
CACHE_DIR = Path(os.getenv("BIOINFO_CACHE_DIR", str(DEFAULT_CACHE_DIR)))

df = pd.read_csv(str(CACHE_DIR / 'augmented_targets.csv'))
    # Run this diagnostic to see what's being dropped
def diagnose_triplet_loss(augmented_df):
        anchors   = augmented_df[augmented_df['type'] == 'noncanonical_target']
        positives = augmented_df[augmented_df['type'] == 'positive_pair']
        negatives = augmented_df[augmented_df['type'] == 'hard_negative_pair']
    
        print(f"Total anchors: {len(anchors)}")
        print(f"Anchors with positives: {positives['anchor_id'].nunique()}")
        print(f"Anchors with negatives: {negatives['anchor_id'].nunique()}")
    
        dropped = []
        for _, row in anchors.iterrows():
            aid      = row['id']
            has_pos  = aid in positives['anchor_id'].values
            has_neg  = aid in negatives['anchor_id'].values
            if not (has_pos and has_neg):
                dropped.append({'id': aid, 'has_pos': has_pos, 'has_neg': has_neg})
    
        print(f"\nDropped anchors ({len(dropped)}):")
        for d in dropped:
            print(f"  {d['id']}: pos={d['has_pos']} neg={d['has_neg']}")
        
diagnose_triplet_loss(df)