from rdkit import Chem
import pandas as pd
from rdkit import Chem
import pandas as pd

def hallucinate_smiles(base_smiles, num_variants=5):
    """Generates Positive Pairs (Synonymous SMILES strings)."""
    mol = Chem.MolFromSmiles(base_smiles)
    if not mol:
        return []

    variants = set()
    for _ in range(num_variants * 10): 
        random_smiles = Chem.MolToSmiles(mol, doRandom=True)
        variants.add(random_smiles)
        if len(variants) == num_variants:
            break
            
    return list(variants)

def generate_hard_negatives(base_smiles):
    """
    The Chemical Scissors. Hunts for electronegative hotspots 
    and surgically mutates them to create biological 'decoys'.
    """
    mol = Chem.MolFromSmiles(base_smiles)
    if not mol:
        return []

    hard_negatives = set()

    # The Mutation Dictionary: [Target SMARTS Pattern] -> [Replacement SMARTS]
    # We are swapping highly reactive atoms for chemically distinct 'decoys'
    mutations = {
    '[F:1]':      '[Cl:1]',   # Safe - F->Cl
    '[Cl:1]':     '[C:1]',    # Safe - Cl->C  
    '[OH:1]':     '[C:1]',    # Mostly safe - OH->C
    '[O:1]':      '[S:1]',    # O->S (CAUTION: may match backbone carbonyls)
    '[NH2:1]':    '[C:1]',    # NH2->C (terminal amines only)
    }

    for smarts_target, replacement_smarts in mutations.items():
        pattern = Chem.MolFromSmarts(smarts_target)
        replacement = Chem.MolFromSmarts(replacement_smarts)

        # Validate that both pattern and replacement are valid
        if pattern is None or replacement is None:
            continue

        # If the molecule contains the highly reactive target atom...
        if mol.HasSubstructMatch(pattern):
            try:
                # ...mathematically cut it out and paste the replacement in
                new_mols = Chem.ReplaceSubstructs(mol, pattern, replacement, replaceAll=False)
                
                for new_mol in new_mols:
                    # Sanity Check: Did this mutation break the laws of physics?
                    try:
                        Chem.SanitizeMol(new_mol)
                        hard_negatives.add(Chem.MolToSmiles(new_mol))
                    except Exception:
                        continue # If valency is broken, throw it away
            except Exception:
                continue # Skip if ReplaceSubstructs fails

    return list(hard_negatives)

def run_augmentation_pipeline(df, pos_factor=5):
    """Expands the dataframe with both Positives and Hard Negatives."""
    print(f"Starting Augmentation Engine...")
    
    augmented_rows = []
    
    for index, row in df.iterrows():
        # 1. Keep the Anchor (Original Data)
        augmented_rows.append(row.to_dict())
        
        if row['type'] == 'noncanonical_target':
            
            # 2. Generate Positive Pairs
            positives = hallucinate_smiles(row['smiles'], num_variants=pos_factor)
            for new_smiles in positives:
                augmented_rows.append({
                    'id': f"{row['id']}_pos",
                    'smiles': new_smiles,
                    'type': 'positive_pair',
                    'anchor_id': row['id'] # Link it back to the original
                })
                
            # 3. Generate Hard Negative Pairs
            negatives = generate_hard_negatives(row['smiles'])
            for new_smiles in negatives:
                augmented_rows.append({
                    'id': f"{row['id']}_neg",
                    'smiles': new_smiles,
                    'type': 'hard_negative_pair',
                    'anchor_id': row['id'] # Link it back to the original
                })
                
    final_df = pd.DataFrame(augmented_rows)
    print(f"Augmentation Complete. Dataset grew from {len(df)} to {len(final_df)} sequences.")
    return final_df