"""
Belka Transformer - Legacy Compatibility Module

This module provides backward compatibility with the original notebook-style implementation
while leveraging the new modular architecture.

For new development, use the modular components directly:
- belka_utils: Custom layers, losses, metrics, and model
- data_processing: Data pipeline functions  
- training: Training and evaluation functions
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Union

# Import from new modular structure
from belka_utils import (
    MultiLabelLoss, CategoricalLoss, BinaryLoss, MaskedAUC,
    FPGenerator, Encodings, Embeddings, FeedForward, SelfAttention, EncoderLayer,
    Belka, load_model
)
from .data_processing import make_parquet, get_vocab, get_smiles_encoder, make_dataset, train_val_datasets
from training import train_model, make_submission


# BACKWARD COMPATIBILITY FUNCTIONS
# These functions maintain the original interface for legacy code

def set_parameters(
        activation: str, batch_size: int, buffer_size: Union[int, float],
        depth: int, dropout_rate: float, epochs: int, epsilon: float, initial_epoch: int,
        masking_rate: float, max_length: int, mode: str, model: Union[str, None],
        model_name: str, num_heads: int, num_layers: int,
        patience: int, root: str, seed: int, steps_per_epoch: int, validation_steps: int, 
        vocab: str, vocab_size: int, working: str) -> dict:
    """
    Legacy function for setting uniform parameters.
    
    Note: This function is kept for backward compatibility.
    For new code, use the configuration system in configs/config.yaml
    """
    return {
        'activation': activation,
        'batch_size': batch_size,
        'buffer_size': int(buffer_size),
        'depth': depth,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'epsilon': epsilon,
        'initial_epoch': initial_epoch,
        'masking_rate': masking_rate,
        'max_length': max_length,
        'mode': mode,
        'model': model,
        'model_name': model_name,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'patience': patience,
        'root': root,
        'seed': seed,
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'working': working
    }


# Legacy dataset creation function
def train_val_set(batch_size: int, buffer_size: int, masking_rate: float, max_length: int, 
                  mode: str, seed: int, vocab_size: int, working: str, **kwargs) -> tuple:
    """
    Legacy wrapper for train_val_datasets function.
    
    Note: This function is kept for backward compatibility.
    Use train_val_datasets from data_processing module for new code.
    """
    return train_val_datasets(
        batch_size=batch_size,
        buffer_size=buffer_size, 
        masking_rate=masking_rate,
        max_length=max_length,
        mode=mode,
        seed=seed,
        vocab_size=vocab_size,
        working=working,
        vocab=kwargs.get('vocab', os.path.join(working, 'vocab.txt')),
        **kwargs
    )


# Example usage with legacy parameters (commented out for safety)
if __name__ == "__main__":
    print("Belka Transformer - Modular Implementation")
    print("=========================================")
    print()
    print("This module now uses a modular architecture.")
    print("For new development, use:")
    print("  - belka_utils: Custom components")
    print("  - data_processing: Data pipeline")  
    print("  - training: Model training")
    print("  - scripts/pipeline.py: CLI interface")
    print()
    print("Example usage:")
    print("  python scripts/pipeline.py --step full --mode clf")
    print("  python scripts/pipeline.py --step train_model --mode mlm")
    print("  python scripts/pipeline.py --step make_submission --model-path model.keras")
    print()
    
    # Legacy parameter example (kept for reference)
    # Initialize mapply for parallel processing
    # mapply.init(n_workers=4, progressbar=True)
    
    # parameters = set_parameters(
    #     root='/pub/ddlin/projects/belka_del/data/raw',
    #     working='/pub/ddlin/projects/belka_del/data/raw',
    #     vocab='/pub/ddlin/projects/belka_del/data/raw/vocab.txt',
    #     model=None,
    #     mode='clf',
    #     model_name='belka',
    #     masking_rate=0.15,
    #     batch_size=2048, buffer_size=1e07,
    #     epochs=1000, initial_epoch=0, steps_per_epoch=10000, validation_steps=2000,
    #     max_length=128, vocab_size=43,
    #     depth=32, dropout_rate=0.1, num_heads=8, num_layers=4, activation='gelu',
    #     patience=20, epsilon=1e-07, seed=42)
    
    # Example function calls:
    # make_parquet_memory_safe(**parameters)  
    # get_vocab(**parameters)
    # make_dataset(**parameters)
    # model = train_model(**parameters)
    # make_submission(**parameters, model_path='path/to/model.keras')