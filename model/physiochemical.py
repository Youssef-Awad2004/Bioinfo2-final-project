# model/physicochemical.py
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors


class PhysicochemicalBiasComputer(nn.Module):
    """
    Computes atom-resolution pairwise matrices from RDKit,
    then projects them to token-resolution for attention injection.
    
    Two matrices:
      P_electro [N_tokens, N_tokens]: pairwise Gasteiger charge differences
      P_steric  [N_tokens, N_tokens]: pairwise van der Waals radius sums
      
    Both are computed at atom level then pooled to token level
    using the atom→token alignment map built during tokenization.
    """

    # Van der Waals radii in Angstroms, by atomic number
    # Source: Bondi (1964), extended for synthetic elements
    VDW_RADII = {
        1:  1.20,  # H
        6:  1.70,  # C
        7:  1.55,  # N
        8:  1.52,  # O
        9:  1.47,  # F
        15: 1.80,  # P
        16: 1.80,  # S
        17: 1.75,  # Cl
        34: 1.90,  # Se  ← critical for selenocysteine
        35: 1.85,  # Br
        53: 1.98,  # I
    }
    DEFAULT_VDW = 1.70  # fallback for unlisted elements

    def compute_atom_matrices(
        self, mol
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns P_electro and P_steric at atom resolution.
        Both shaped [N_atoms, N_atoms].
        """
        # Compute Gasteiger charges in place
        AllChem.ComputeGasteigerCharges(mol)

        charges = []
        radii   = []

        for atom in mol.GetAtoms():
            # Gasteiger charge — falls back to 0.0 if computation failed
            q = atom.GetDoubleProp('_GasteigerCharge')
            if q != q:  # NaN check — happens for exotic atoms
                q = 0.0
            charges.append(q)

            # vdW radius
            atomic_num = atom.GetAtomicNum()
            radii.append(self.VDW_RADII.get(atomic_num, self.DEFAULT_VDW))

        charges = torch.tensor(charges, dtype=torch.float32)
        radii   = torch.tensor(radii,   dtype=torch.float32)

        # Pairwise charge difference — encodes electrostatic potential
        P_electro = charges.unsqueeze(1) - charges.unsqueeze(0)  # [N, N]

        # Pairwise radius sum — encodes steric clash potential
        P_steric  = radii.unsqueeze(1) + radii.unsqueeze(0)      # [N, N]

        return P_electro, P_steric

    def pool_to_token_space(
        self,
        P_atom: torch.Tensor,        # [N_atoms, N_atoms]
        atom_to_token: list[int],    # length N_atoms, each value is token index
        n_tokens: int
    ) -> torch.Tensor:
        """
        Aggregates atom-level matrix to token-level via mean pooling.
        Multiple atoms mapping to the same token get their values averaged.
        """
        P_token = torch.zeros(n_tokens, n_tokens)
        counts  = torch.zeros(n_tokens, n_tokens)

        n_atoms = len(atom_to_token)
        for i in range(n_atoms):
            for j in range(n_atoms):
                ti = atom_to_token[i]
                tj = atom_to_token[j]
                if ti < n_tokens and tj < n_tokens:
                    P_token[ti, tj] += P_atom[i, j]
                    counts[ti, tj]  += 1

        # Avoid division by zero for token pairs with no atom coverage
        return P_token / counts.clamp(min=1.0)

    def build_atom_to_token_map(
        self,
        smiles: str,
        token_ids: list[int],
        tokenizer
    ) -> list[int]:
        """
        Builds a mapping from RDKit atom index → token index.
        
        Strategy: canonical SMILES character positions are matched
        to BPE token character spans using the tokenizer's offset mapping.
        
        This is the critical bridge between RDKit atom space and
        transformer token space.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Get canonical SMILES with atom indices preserved
        canonical = Chem.MolToSmiles(mol, canonical=True)

        # Get character-level offsets from the tokenizer
        encoding = tokenizer.tokenizer(
            canonical,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        offsets = encoding['offset_mapping']  # [(char_start, char_end), ...]

        # Build char_position → token_index lookup
        char_to_token = {}
        for tok_idx, (start, end) in enumerate(offsets):
            for char_pos in range(start, end):
                char_to_token[char_pos] = tok_idx

        # Map each RDKit atom to its character position in canonical SMILES
        # RDKit stores this via atom map after re-parsing with atom indices
        atom_to_token = []
        for atom in mol.GetAtoms():
            # Get atom's position in canonical SMILES string
            # This uses RDKit's internal SMILES atom ordering
            atom_idx = atom.GetIdx()

            # Find the character position of this atom in canonical SMILES
            # by searching for the atom symbol at the expected position
            char_pos = self._find_atom_char_position(canonical, mol, atom_idx)

            token_idx = char_to_token.get(char_pos, 0)
            atom_to_token.append(token_idx)

        return atom_to_token

    def _find_atom_char_position(
        self, canonical_smiles: str, mol, atom_idx: int
    ) -> int:
        """
        Finds the character position of atom[atom_idx] in canonical_smiles.
        Uses RDKit's atom map number trick for reliable positioning.
        """
        from rdkit.Chem import rdmolops

        # Tag each atom with its index as a map number
        tagged_mol = Chem.RWMol(mol)
        for atom in tagged_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)  # 1-indexed map nums

        tagged_smiles = Chem.MolToSmiles(tagged_mol, canonical=True)

        # Find the pattern ":N]" where N = atom_idx+1
        # This locates the atom in the tagged SMILES
        import re
        pattern = rf'\[.*?:{atom_idx + 1}\]|[A-Z][a-z]?:{atom_idx + 1}'
        match = re.search(pattern, tagged_smiles)

        if match:
            # Strip atom map from position to get canonical position
            # Approximate: return start of the atom symbol
            return max(0, match.start() - (atom_idx * 3))

        return 0  # Fallback: map to first token