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
from dataPipeline.precompute_physics import ExactPhysicsLookup


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
        physics_lookup: dict,
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
        self.triplets = self._build_triplets(augmented_df)
        self.physics = ExactPhysicsLookup(physics_cache, max_length)

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

    def _encode_one(self, smiles: str) -> dict:
        token_ids = torch.tensor(
            self.tokenizer.encode_smiles(smiles, max_length=self.max_length),
            dtype=torch.long
        )
        # Single dictionary lookup — zero RDKit
        P_e, P_s = self.physics.get(smiles)

        return {
            'token_ids': token_ids,
            'P_electro': P_e,
            'P_steric':  P_s,
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
    



    # In dataset.py — add this class

class PhysicsLookup:
    """
    Wraps the precomputed physics lookup table.
    Handles the atom-to-token projection at lookup time,
    which must happen per SMILES variant (not precomputable).
    """

    def __init__(self, lookup: dict, max_length: int = 256):
        self.lookup     = lookup
        self.max_length = max_length

    def get(
        self,
        smiles: str,
        token_ids: list[int],
        tokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Looks up precomputed atom-level matrices and projects to token space.
        
        The atom-level matrices are precomputed and cached.
        The atom-to-token projection is computed here because it depends
        on the specific SMILES variant's tokenization — not precomputable.
        
        Returns P_electro and P_steric at token resolution, padded to max_length.
        """
        T   = self.max_length
        mol = Chem.MolFromSmiles(smiles)

        # Look up by canonical SMILES
        canonical = Chem.MolToSmiles(mol) if mol else None
        entry     = self.lookup.get(canonical) if canonical else None

        if entry is None:
            # Molecule not in lookup — return zeros
            # This should not happen if lookup was built from the same dataset
            return torch.zeros(T, T), torch.zeros(T, T)

        P_e_atom = entry['P_electro']   # [N_atoms, N_atoms]
        P_s_atom = entry['P_steric']    # [N_atoms, N_atoms]
        n_atoms  = entry['n_atoms']

        # Project atom matrices to token space
        # This must be done per SMILES variant because token alignment varies
        P_e_token, P_s_token = self._project_to_token_space(
            P_e_atom, P_s_atom, n_atoms, token_ids, T
        )

        return P_e_token, P_s_token

    def _project_to_token_space(
        self,
        P_e_atom: torch.Tensor,
        P_s_atom: torch.Tensor,
        n_atoms: int,
        token_ids: list[int],
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects atom-resolution matrices to token-resolution via
        linear index scaling. Fast O(n_atoms^2) operation — no RDKit calls.
        """
        # Count real (non-padding) tokens
        pad_id    = 1   # your pad_token_id
        n_tokens  = sum(1 for t in token_ids if t != pad_id)
        n_tokens  = max(n_tokens, 1)

        P_e = torch.zeros(T, T)
        P_s = torch.zeros(T, T)

        if n_atoms == 0:
            return P_e, P_s

        # Linear scaling from atom indices to token indices
        # This is an approximation — exact atom-to-token alignment
        # requires offset mapping which is expensive at training time.
        # Linear scaling is fast and sufficient for the physics signal.
        scale = n_tokens / n_atoms

        for i in range(n_atoms):
            ti = min(int(i * scale), T - 1)
            for j in range(n_atoms):
                tj = min(int(j * scale), T - 1)
                P_e[ti, tj] = P_e_atom[i, j]
                P_s[ti, tj] = P_s_atom[i, j]

        return P_e, P_s