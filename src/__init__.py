"""
Belka Transformer Package

A modular molecular transformer pipeline for binding affinity prediction.
"""

from .belka_utils import (
    # Custom losses
    MultiLabelLoss,
    CategoricalLoss, 
    BinaryLoss,
    
    # Custom metrics
    MaskedAUC,
    
    # Custom layers
    FPGenerator,
    Encodings,
    Embeddings,
    FeedForward,
    SelfAttention,
    EncoderLayer,
    
    # Model
    Belka,
    load_model
)

from .data_processing import (
    read_parquet,
    make_parquet,
    make_parquet_memory_safe,
    get_vocab,
    get_smiles_encoder,
    make_dataset,
    create_validation_dataset,
    train_val_datasets,
    initialize_mapply
)

from .training import (
    create_model,
    setup_callbacks,
    train_model,
    evaluate_model,
    make_submission,
    predict_molecules,
    save_training_config,
    load_training_config
)

__version__ = "1.0.0"
__author__ = "Belka Transformer Team"
__description__ = "Molecular transformer for binding affinity prediction"