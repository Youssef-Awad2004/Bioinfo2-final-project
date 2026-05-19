# train/dataset.py
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from model.physiochemical import PhysicochemicalBiasComputer


class MolecularTripletDataset(Dataset):
    """
    Returns (anchor, positive, hard_neg, bg_neg) quadruplets.
    Each element contains:
        - token_ids:  [seq_len]        integer tensor
        - P_electro:  [seq_len, seq_len] float tensor
        - P_steric:   [seq_len, seq_len] float tensor
    """

    def __init__(
        self,
        augmented_df,
        canonical_df,
        tokenizer,
        max_length: int = 128,
    ):
        self.tokenizer      = tokenizer
        self.max_length     = max_length
        self.physics        = PhysicochemicalBiasComputer()
        self.canonical_pool = canonical_df['smiles'].dropna().tolist()
        self._morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )
        self.triplets       = self._build_triplets(augmented_df)

    def _sample_background_negative(self, anchor_smiles: str, max_attempts: int = 50) -> str:
        """
        Sample a canonical baseline that stays chemically distant from the anchor.

        Rejects candidates with Tanimoto similarity > 0.6 against the anchor.
        Falls back to the last sampled canonical if no distant candidate is found.
        """
        anchor_mol = Chem.MolFromSmiles(anchor_smiles)
        if anchor_mol is None or not self.canonical_pool:
            return anchor_smiles

        anchor_fp = self._morgan_generator.GetFingerprint(anchor_mol)
        chosen_smiles = self.canonical_pool[0]

        for _ in range(max_attempts):
            candidate_smiles = self.canonical_pool[
                torch.randint(len(self.canonical_pool), (1,)).item()
            ]
            candidate_mol = Chem.MolFromSmiles(candidate_smiles)
            if candidate_mol is None:
                continue

            candidate_fp = self._morgan_generator.GetFingerprint(candidate_mol)
            tanimoto = DataStructs.TanimotoSimilarity(anchor_fp, candidate_fp)
            chosen_smiles = candidate_smiles

            if tanimoto <= 0.6:
                return candidate_smiles

        return chosen_smiles

    def _build_triplets(self, df):
        anchors   = df[df['type'] == 'noncanonical_target']
        positives = df[df['type'] == 'positive_pair']
        negatives = df[df['type'] == 'hard_negative_pair']

        triplets = []
        for _, anchor_row in anchors.iterrows():
            aid      = anchor_row['id']
            pos_pool = positives[positives['anchor_id'] == aid]['smiles'].tolist()
            neg_pool = negatives[negatives['anchor_id'] == aid]['smiles'].tolist()

            if pos_pool and neg_pool:
                triplets.append({
                    'anchor':   anchor_row['smiles'],
                    'positive': pos_pool,   # keep full pool — sample at __getitem__
                    'negative': neg_pool,
                })

        print(f"Built {len(triplets)} valid triplets from dataset")
        return triplets

    def _encode_one(self, smiles: str) -> dict[str, torch.Tensor]:
        """
        Full encoding pipeline for a single SMILES string.
        Returns token_ids, P_electro, P_steric all at token resolution.
        """
        T = self.max_length

        # 1. Tokenize
        token_ids = torch.tensor(
            self.tokenizer.encode_smiles(smiles, max_length=T),
            dtype=torch.long
        )

        # 2. Compute physicochemical matrices
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Return zero matrices for invalid SMILES — should not happen
            # after sanitization in augmenter, but defensive coding matters
            return {
                'token_ids': token_ids,
                'P_electro': torch.zeros(T, T),
                'P_steric':  torch.zeros(T, T),
            }

        P_e_atom, P_s_atom = self.physics.compute_atom_matrices(mol)

        # 3. Project atom-resolution matrices to token-resolution
        atom_to_token = self.physics.build_atom_to_token_map(
            smiles, token_ids.tolist(), self.tokenizer
        )

        n_real_tokens = (token_ids != self.tokenizer.pad_token_id).sum().item()

        P_e_token = self.physics.pool_to_token_space(
            P_e_atom, atom_to_token, n_real_tokens
        )
        P_s_token = self.physics.pool_to_token_space(
            P_s_atom, atom_to_token, n_real_tokens
        )

        # 4. Pad to max_length × max_length
        P_e_padded = torch.zeros(T, T)
        P_s_padded = torch.zeros(T, T)
        n = min(n_real_tokens, T)
        P_e_padded[:n, :n] = P_e_token[:n, :n]
        P_s_padded[:n, :n] = P_s_token[:n, :n]

        return {
            'token_ids': token_ids,
            'P_electro': P_e_padded,
            'P_steric':  P_s_padded,
        }

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]

        # Sample one positive and one negative randomly at fetch time
        # This gives you different augmentation variants each epoch
        pos_smiles = triplet['positive'][
            torch.randint(len(triplet['positive']), (1,)).item()
        ]
        neg_smiles = triplet['negative'][
            torch.randint(len(triplet['negative']), (1,)).item()
        ]
        bg_smiles = self._sample_background_negative(triplet['anchor'])

        return {
            'anchor':   self._encode_one(triplet['anchor']),
            'positive': self._encode_one(pos_smiles),
            'hard_neg': self._encode_one(neg_smiles),
            'bg_neg':   self._encode_one(bg_smiles),
        }