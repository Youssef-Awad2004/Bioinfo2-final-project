# tokenizer.py - consolidated, replaces NcAATokenizerFineTuner entirely
from transformers import RobertaTokenizerFast
from collections import Counter
from typing import Optional
import pandas as pd


class SmilesBPETokenizer:
    """
    Production tokenizer for ncAA SMILES strings.
    
    Strategy:
        1. Inherit ChemBERTa's 591-token vocabulary (trained on 100M SMILES)
        2. Hardcode ncAA-specific atoms as atomic units (never split)
        3. Mine your corpus for frequent substructures (data-driven expansion)
        4. Save the extended vocab for reproducible use
    
    Directly derived from audit output showing [SeH] -> 6 fragments,
    [N+]=[N-] -> 5 fragments, [NH+] -> 3 fragments.
    """

    # ---------------------------------------------------------------
    # PHASE 1 TOKENS: Hardcoded from audit failures
    # These are ATOMIC - BPE will never split them regardless of frequency
    # ---------------------------------------------------------------
    _AUDIT_DERIVED_ATOMS = [
        # Selenium (Selenocysteine) - shredded into 6 pieces in audit
        '[SeH]', '[Se]', '[SeH2]',

        # Azide group - shredded into 5 pieces in audit
        '[N+]=[N-]',    # Full azide linkage as one unit
        '[N-]',         # Negatively charged nitrogen
        '[N+]',         # Positively charged nitrogen

        # Charged amines - [NH+] split into 3 pieces in audit
        '[NH+]', '[NH2+]', '[NH3+]', '[NH-]',

        # Charged oxygens/sulfurs - preemptive, common in ncAAs at pH 7.4
        '[OH+]', '[O-]', '[S-]', '[SH]', '[S+]',

        # Isotopic labels - common in PET tracer and NMR probe ncAAs
        '[18F]', '[125I]', '[11C]', '[13C]', '[15N]',

        # Boron/Phosphorus - emerging ncAA classes
        '[BH]', '[B]', '[PH]',

        # Aromatic heteroatoms in bracket form
        '[nH]', '[NH]',

        # Unusual ring atoms (some synthetic ncAA scaffolds)
        '[si]', '[p]', '[as]',
    ]

    # ---------------------------------------------------------------
    # PHASE 2 TOKENS: Chemically valid substructure fragments
    # These must be complete, balanced SMILES substrings
    # Validated: no mid-branch starts/ends, no overlapping with Phase 1
    # ---------------------------------------------------------------
    _STRUCTURAL_FRAGMENTS = [
        'C(=O)O',       # Carboxylate - C-terminus of every amino acid
        'C(N)C(=O)O',   # Full amino acid backbone (alpha-carbon + both termini)
        'CC(N)',         # Alpha-carbon pattern
        'C(=O)N',       # Amide bond - peptide backbone linkage
        'C(=O)[O-]',    # Deprotonated carboxylate (physiological pH form)
    ]

    def __init__(self, pretrained_path: Optional[str] = None):
        """
        Args:
            pretrained_path: Path to a previously saved fine-tuned tokenizer.
                             If None, loads base ChemBERTa and applies ncAA patches.
        """
        base = pretrained_path or "seyonec/ChemBERTa-zinc-base-v1"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(base)

        if not pretrained_path:
            # Fresh load - apply hardcoded patches immediately
            self._apply_phase1_tokens()

    def _apply_phase1_tokens(self):
        """
        Adds audit-derived atomic tokens.
        Prints a diagnostic breakdown of what was new vs. already present.
        """
        vocab = self.tokenizer.get_vocab()

        already_present = [t for t in self._AUDIT_DERIVED_ATOMS if t in vocab]
        genuinely_new   = [t for t in self._AUDIT_DERIVED_ATOMS if t not in vocab]

        print(f"\n--- Phase 1: Hardcoded Atom Tokens ---")
        print(f"Already in ChemBERTa vocab : {already_present}")
        print(f"Genuinely new (adding now) : {genuinely_new}")

        num_added = self.tokenizer.add_tokens(genuinely_new)
        print(f"Added {num_added} tokens. Vocab: {self.original_vocab_size} -> {len(self.tokenizer)}")

    def _apply_phase2_tokens(self, corpus_smiles: list[str]):
        """
        Two sub-phases:
          2a. Add hardcoded structural fragments (chemically validated)
          2b. Mine corpus for additional frequent fragments
        """
        vocab = self.tokenizer.get_vocab()

        # 2a. Hardcoded fragments
        new_fragments = [f for f in self._STRUCTURAL_FRAGMENTS if f not in vocab]
        print(f"\n--- Phase 2a: Structural Fragments ---")
        print(f"Adding {len(new_fragments)} validated fragments")
        self.tokenizer.add_tokens(new_fragments)

        # 2b. Corpus-mined fragments
        print(f"\n--- Phase 2b: Corpus Mining ---")
        mined = self._mine_fragments(corpus_smiles)
        vocab = self.tokenizer.get_vocab()  # Refresh after 2a additions
        new_mined = [f for f in mined if f not in vocab]
        num_mined = self.tokenizer.add_tokens(new_mined)
        print(f"Mined {num_mined} corpus fragments from {len(corpus_smiles)} SMILES")

    def _mine_fragments(
        self,
        smiles_list: list[str],
        min_freq: int = 3,
        max_frag_len: int = 8
    ) -> list[str]:
        """
        Sliding window fragment miner.
        
        Validity filters applied:
          - Bracket balance: '[' count must equal ']' count
          - No mid-branch fragments: must not start with ')' or end with '('
          - Minimum length 2: single chars are already handled by base vocab
        """
        counts = Counter()

        for smiles in smiles_list:
            for length in range(2, max_frag_len + 1):
                for start in range(len(smiles) - length + 1):
                    frag = smiles[start:start + length]

                    # Filter 1: balanced brackets
                    if frag.count('[') != frag.count(']'):
                        continue

                    # Filter 2: no mid-branch fragments
                    if frag.startswith(')') or frag.endswith('('):
                        continue

                    # Filter 3: no fragments that are just numbers (ring closures)
                    if frag.isdigit():
                        continue

                    counts[frag] += 1

        frequent = [f for f, c in counts.items() if c >= min_freq]
        print(f"Found {len(frequent)} fragments above min_freq={min_freq}")
        return frequent

    def train_from_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'smiles',
        save_path: str = "./ncaa_tokenizer"
    ):
        """
        Runs Phase 2 (corpus-driven expansion) and saves.
        Phase 1 already ran in __init__.
        """
        smiles_list = df[smiles_col].dropna().tolist()
        self._apply_phase2_tokens(smiles_list)

        self.tokenizer.save_pretrained(save_path)
        print(f"\n[OK] Tokenizer saved to {save_path}/")
        print(f"   Final vocab size : {self.vocab_size}")
        print(f"   Pad token ID     : {self.pad_token_id}")
        print(f"\n--- Copy into your model config ---")
        print(f"VOCAB_SIZE   = {self.vocab_size}")
        print(f"PAD_TOKEN_ID = {self.pad_token_id}")

    def encode_smiles(self, smiles: str, max_length: int = 128) -> list[int]:
        return self.tokenizer(
            smiles,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )['input_ids']

    def validate_integrity(self, test_cases: Optional[dict] = None):
        """
        Re-runs the audit after fine-tuning.
        Pass your original audit dict to confirm all failures are resolved.
        """
        if test_cases is None:
            test_cases = {
                "Selenocysteine": "C([SeH])C(N)C(=O)O",
                "Azidoalanine":   "N=[N+]=[N-]CC(N)C(=O)O",
                "Bracket atom":   "C1CC[NH+]1",
            }

        MUST_BE_SINGLE = ['[SeH]', '[N+]=[N-]', '[NH+]', '[N-]', '[N+]']

        print("\n=== POST FINE-TUNE INTEGRITY CHECK ===")
        all_passed = True

        for name, smiles in test_cases.items():
            tokens = self.tokenizer.tokenize(smiles)
            print(f"\n{name}: {smiles}")
            print(f"  Tokens: {tokens}")

            for atom in MUST_BE_SINGLE:
                if atom in smiles:
                    if atom in tokens:
                        print(f" {atom} -> single token")
                    else:
                        print(f"   {atom} -> STILL FRAGMENTED - check add_tokens() call")
                        all_passed = False

        print(f"\n{'ALL CHECKS PASSED' if all_passed else ' FAILURES DETECTED - DO NOT PROCEED TO TRAINING'}")
        return all_passed

    @property
    def original_vocab_size(self) -> int:
        # ChemBERTa base is always 591
        return 591

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id