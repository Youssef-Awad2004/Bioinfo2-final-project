# model/weight_transfer.py
import torch
from transformers import RobertaModel


def transfer_chemberta_weights(
    your_model: 'NcAATransformerEncoder',
    chemberta_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    verbose: bool = True
) -> 'NcAATransformerEncoder':
    """
    Surgically transfers ChemBERTa-2 pretrained weights into your model.
    
    What gets transferred:
        - Token embeddings (first 591 rows — shared vocabulary)
        - Attention Q/K/V/output projections (all 6 layers)
        - FFN intermediate and output projections (all 6 layers)
        - LayerNorm gamma and beta (all layers)
        
    What stays randomly initialized:
        - Token embedding rows 591-787 (your new ncAA tokens)
        - alpha, W_e, W_s in each attention layer (new physics gate)
        - AttentionWeightedPooling query vector
        - Projection head (entirely new)
    """
    print(f"Loading ChemBERTa weights from {chemberta_name}...")
    chemberta = RobertaModel.from_pretrained(chemberta_name)
    chembert_state = chemberta.state_dict()

    transferred = []
    skipped     = []

    # ── 1. Token Embeddings ─────────────────────────────────────────────
    # ChemBERTa vocab: 591 tokens. Your vocab: 788 tokens.
    # Copy the first 591 rows. Rows 591-787 keep Xavier initialization.
    chembert_embed = chembert_state['embeddings.word_embeddings.weight']
    n_pretrained   = chembert_embed.shape[0]      # 591
    n_yours        = your_model.embedding.weight.shape[0]  # 788

    with torch.no_grad():
        your_model.embedding.weight[:n_pretrained] = chembert_embed
    transferred.append(f'embedding [{n_pretrained}/{n_yours} rows]')

    # ── 2. Positional Embeddings ────────────────────────────────────────
    chembert_pos = chembert_state['embeddings.position_embeddings.weight']
    with torch.no_grad():
        # ChemBERTa uses 514 positions. Take first max_seq_len+2.
        n_pos = min(chembert_pos.shape[0], your_model.pos_embedding.weight.shape[0])
        your_model.pos_embedding.weight[:n_pos] = chembert_pos[:n_pos]
    transferred.append(f'pos_embedding [{n_pos} positions]')

    # ── 3. Transformer Layer Weights ────────────────────────────────────
    # ChemBERTa has 6 layers — same as your model.
    # RoBERTa naming: encoder.layer.{i}.attention.self.{q,k,v}_proj
    # Your naming:    layers.{i}.attention.{q,k,v}_proj
    
    layer_map = {
        # ChemBERTa key pattern → your model key pattern
        'encoder.layer.{i}.attention.self.query': 'layers.{i}.attention.q_proj',
        'encoder.layer.{i}.attention.self.key':   'layers.{i}.attention.k_proj',
        'encoder.layer.{i}.attention.self.value':  'layers.{i}.attention.v_proj',
        'encoder.layer.{i}.attention.output.dense': 'layers.{i}.attention.out_proj',
        'encoder.layer.{i}.intermediate.dense':    'layers.{i}.ffn.0',
        'encoder.layer.{i}.output.dense':          'layers.{i}.ffn.3',
        'encoder.layer.{i}.attention.output.LayerNorm': 'layers.{i}.norm1',
        'encoder.layer.{i}.output.LayerNorm':      'layers.{i}.norm2',
    }

    for layer_idx in range(6):
        for chembert_pattern, your_pattern in layer_map.items():
            chembert_key = chembert_pattern.replace('{i}', str(layer_idx))
            your_key     = your_pattern.replace('{i}', str(layer_idx))

            # Transfer weight and bias separately
            for param in ['weight', 'bias']:
                src_key = f'{chembert_key}.{param}'
                dst_key = f'{your_key}.{param}'

                if src_key in chembert_state:
                    src_tensor = chembert_state[src_key]
                    dst_param  = dict(your_model.named_parameters()).get(dst_key)

                    if dst_param is not None and dst_param.shape == src_tensor.shape:
                        with torch.no_grad():
                            dst_param.copy_(src_tensor)
                        transferred.append(f'layer{layer_idx}.{your_key}.{param}')
                    else:
                        skipped.append(f'SHAPE MISMATCH: {dst_key}')
                else:
                    skipped.append(f'NOT FOUND: {src_key}')

    if verbose:
        print(f"\n✅ Transferred {len(transferred)} parameter tensors")
        print(f"⚠️  Skipped {len(skipped)} (new params — will train from scratch)")
        if skipped:
            print(f"   Skipped keys: {skipped[:5]}{'...' if len(skipped)>5 else ''}")

    # Clean up — free ChemBERTa from memory
    del chemberta, chembert_state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return your_model