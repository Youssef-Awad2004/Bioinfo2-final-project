# model/config.py
import os
from pathlib import Path

def load_model_config(tokenizer_path: str = None) -> dict:
    if tokenizer_path is None:
        PROJECT_DIR = Path(__file__).resolve().parent.parent
        tokenizer_path = str(Path(os.getenv("BIOINFO_TOKENIZER_DIR", str(PROJECT_DIR / "ncaa_tokenizer"))))
    
    from transformers import RobertaTokenizerFast
    tok = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    
    return {
        'vocab_size':     len(tok),
        'pad_token_id':   tok.pad_token_id,
        'embed_dim':      768,    # ← changed from 256 to match ChemBERTa
        'num_heads':      12,     # ← changed from 8 (768/12=64 per head, same as ChemBERTa)
        'num_layers':     6,
        'ffn_dim':        3072,   # ← changed from 1024 (ChemBERTa uses 768*4)
        'max_seq_len':    256,    # ← increased from 128 to accommodate longer SMILES
        'dropout':        0.1,
        'projection_dim': 128,    # ← bumped from 64, more reasonable for 768-dim input
    }

TRAINING_CONFIG = {
    'lr':             1e-5,
    'weight_decay':   0.05,
    'tau_max':        0.5,
    'tau_min':        0.1,
    'anneal_steps':   10000,
    'batch_size':     16,         # ← reduced from 32, 768-dim uses more memory
    'epochs':         50,
    'early_stop_patience': 5,     # stop if val gap doesn't improve for 5 evals
    'eval_every':          2,     # evaluate every 2 epochs, not 5
    'min_delta':           0.01,  # minimum improvement to count as progress
}