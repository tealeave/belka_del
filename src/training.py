"""
Belka Training and Evaluation

This module contains functions for model training, evaluation, and submission generation
for the Belka molecular transformer pipeline.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Union, Dict, Any
import atomInSmiles
import logging
import psutil
import time

from .belka_utils import (
    Belka, MultiLabelLoss, CategoricalLoss, BinaryLoss, MaskedAUC, load_model
)
from .data_processing import train_val_datasets, get_smiles_encoder

logger = logging.getLogger(__name__)


def create_model(
    mode: str,
    vocab_size: int,
    activation: str = 'gelu',
    depth: int = 32,
    dropout_rate: float = 0.1,
    epsilon: float = 1e-7,
    max_length: int = 128,
    num_heads: int = 8,
    num_layers: int = 4,
    **kwargs) -> keras.Model:
    """
    Create and compile a Belka transformer model.
    
    Args:
        mode: Training mode ('mlm', 'fps', or 'clf')
        vocab_size: Size of the vocabulary
        activation: Activation function for feed-forward layers
        depth: Hidden dimension size
        dropout_rate: Dropout rate
        epsilon: Small constant for numerical stability
        max_length: Maximum sequence length
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        
    Returns:
        Compiled Keras model
    """
    
    # Create model
    model = Belka(
        activation=activation,
        depth=depth,
        dropout_rate=dropout_rate,
        epsilon=epsilon,
        max_length=max_length,
        mode=mode,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size)
    
    # Configure loss and metrics based on mode
    if mode == 'mlm':
        loss = CategoricalLoss(
            epsilon=epsilon,
            mask=-1,
            vocab_size=vocab_size)
        metrics = MaskedAUC(
            mode=mode,
            mask=-1,
            multi_label=False,
            num_labels=None,
            vocab_size=vocab_size)
    elif mode == 'fps':
        loss = BinaryLoss()
        metrics = MaskedAUC(
            mode=mode,
            mask=-1,
            multi_label=False,
            num_labels=None,
            vocab_size=vocab_size)
    else:  # clf mode
        loss = MultiLabelLoss(
            epsilon=epsilon,
            macro=True,
            nan_mask=2)
        metrics = MaskedAUC(
            mode=mode,
            mask=2,
            multi_label=True,
            num_labels=3,
            vocab_size=vocab_size)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics)
    
    return model


class HPCProgressCallback(keras.callbacks.Callback):
    """HPC-friendly progress tracking callback with detailed logging."""
    
    def __init__(self, log_every_n_batches=100, **kwargs):
        super().__init__(**kwargs)
        self.log_every_n_batches = log_every_n_batches
        self.start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logger.info("=== Training Started ===")
        self._log_system_resources("Training Start")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1} started")
        self._log_system_resources(f"Epoch {epoch + 1} Start")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        total_duration = time.time() - self.start_time
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s (Total: {total_duration:.2f}s)")
        
        if logs:
            for metric, value in logs.items():
                logger.info(f"  {metric}: {value:.6f}")
                
        self._log_system_resources(f"Epoch {epoch + 1} End")
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_every_n_batches == 0:
            logger.info(f"Batch {batch}: {logs}")
            
    def _log_system_resources(self, context):
        """Log system resource usage."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        logger.info(f"System Resources [{context}]:")
        logger.info(f"  Memory: {memory.used / (1024**3):.2f}GB/{memory.total / (1024**3):.2f}GB ({memory.percent:.1f}%)")
        logger.info(f"  CPU: {cpu_percent:.1f}%")
        
        # Log GPU memory if available
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                current_gb = gpu_details['current'] / (1024**3)
                peak_gb = gpu_details['peak'] / (1024**3)
                logger.info(f"  GPU Memory: {current_gb:.2f}GB current, {peak_gb:.2f}GB peak")
        except Exception:
            pass  # GPU monitoring not available

def setup_callbacks(
    mode: str,
    model_name: str,
    working: str,
    patience: int = 20,
    monitor: str = 'loss',
    **kwargs) -> list:
    """
    Setup training callbacks for model training.
    
    Args:
        mode: Training mode ('mlm', 'fps', or 'clf')
        model_name: Base name for saved models
        working: Working directory for saving models
        patience: Early stopping patience
        monitor: Metric to monitor for early stopping
        
    Returns:
        List of configured callbacks
    """
    
    # Define filename suffix based on mode
    suffix_map = {
        'mlm': '_{epoch:03d}_{loss:.4f}.model.keras',
        'fps': '_{epoch:03d}_{auc:.4f}_{val_auc:.4f}.model.keras',
        'clf': '_{epoch:03d}_{auc:.4f}_{val_auc:.4f}.model.keras'
    }
    
    filepath = os.path.join(working, model_name + suffix_map[mode])
    
    # Model checkpoint callback
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode='min' if 'loss' in monitor else 'max',
        filepath=filepath,
        save_best_only=False,
        save_weights_only=False,
        verbose=1)
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode='min' if 'loss' in monitor else 'max',
        patience=patience,
        restore_best_weights=True,
        verbose=1)
    
    # Learning rate reduction callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1)
    
    # Add HPC progress tracking callback
    hpc_progress = HPCProgressCallback(log_every_n_batches=kwargs.get('log_every_n_batches', 100))
    
    return [model_checkpoint, early_stopping, reduce_lr, hpc_progress]


def train_model(
    mode: str,
    model_name: str,
    working: str,
    vocab: str,
    model_path: Union[str, None] = None,
    epochs: int = 1000,
    initial_epoch: int = 0,
    steps_per_epoch: int = 10000,
    validation_steps: int = 2000,
    patience: int = 20,
    batch_size: int = 2048,
    buffer_size: int = 10000000,
    masking_rate: float = 0.15,
    max_length: int = 128,
    vocab_size: int = 43,
    seed: int = 42,
    **kwargs) -> keras.Model:
    """
    Train a Belka transformer model.
    
    Args:
        mode: Training mode ('mlm', 'fps', or 'clf')
        model_name: Base name for saved models
        working: Working directory
        vocab: Path to vocabulary file
        model_path: Path to existing model (for resuming training)
        epochs: Maximum number of training epochs
        initial_epoch: Starting epoch (for resumed training)
        steps_per_epoch: Number of steps per epoch
        validation_steps: Number of validation steps
        patience: Early stopping patience
        batch_size: Training batch size
        buffer_size: Shuffle buffer size
        masking_rate: MLM masking rate
        max_length: Maximum sequence length
        vocab_size: Vocabulary size
        seed: Random seed
        
    Returns:
        Trained model
    """
    
    print(f"Starting training in {mode} mode...")
    
    # Create datasets
    print("Creating datasets...")
    train_ds, val_ds = train_val_datasets(
        batch_size=batch_size,
        buffer_size=buffer_size,
        masking_rate=masking_rate,
        max_length=max_length,
        mode=mode,
        seed=seed,
        vocab_size=vocab_size,
        working=working,
        vocab=vocab,
        **kwargs)
    
    # Create or load model
    if model_path is not None:
        print(f"Loading existing model from: {model_path}")
        model = load_model(model_path)
    else:
        print("Creating new model...")
        model = create_model(
            mode=mode,
            vocab_size=vocab_size,
            max_length=max_length,
            **kwargs)
    
    # Setup callbacks
    callbacks = setup_callbacks(
        mode=mode,
        model_name=model_name,
        working=working,
        patience=patience,
        **kwargs)
    
    # Print model summary
    print("\nModel Architecture:")
    try:
        # Get a sample batch to build the model
        sample_batch = next(iter(train_ds))
        if isinstance(sample_batch, tuple):
            x_sample = sample_batch[0]
        else:
            x_sample = sample_batch
        
        # Call model to build it
        _ = model(x_sample[:1])  # Use first sample to build
        print(model.summary())
    except Exception as e:
        print(f"Could not display model summary: {e}")
    
    # Determine validation settings
    if mode == 'mlm':
        validation_data = None
        validation_steps = None
        print("MLM mode: training without validation")
    else:
        validation_data = val_ds
        print(f"Training with validation: {validation_steps} steps")
    
    print(f"\nStarting training:")
    print(f"- Epochs: {epochs} (starting from {initial_epoch})")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Batch size: {batch_size}")
    print(f"- Mode: {mode}")
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1)
    
    print("Training completed!")
    return model


def evaluate_model(
    model: keras.Model,
    working: str,
    vocab: str,
    mode: str,
    batch_size: int = 2048,
    max_length: int = 128,
    **kwargs) -> Dict[str, float]:
    """
    Evaluate a trained model on the validation set.
    
    Args:
        model: Trained Keras model
        working: Working directory
        vocab: Path to vocabulary file
        mode: Model mode ('mlm', 'fps', or 'clf')
        batch_size: Evaluation batch size
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of evaluation metrics
    """
    
    print("Evaluating model...")
    
    # Create validation dataset
    _, val_ds = train_val_datasets(
        mode=mode,
        working=working,
        vocab=vocab,
        batch_size=batch_size,
        max_length=max_length,
        buffer_size=1000,  # Small buffer for evaluation
        **kwargs)
    
    if val_ds is None:
        print("No validation data available for this mode")
        return {}
    
    # Evaluate model
    results = model.evaluate(val_ds, verbose=1, return_dict=True)
    
    print("Evaluation results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    return results


def make_submission(
    model_path: str,
    working: str,
    vocab: str,
    root: str,
    batch_size: int = 1024,
    max_length: int = 128,
    **kwargs) -> None:
    """
    Generate competition submission using a trained model.
    
    Args:
        model_path: Path to trained model
        working: Working directory
        vocab: Path to vocabulary file
        root: Root directory containing test data
        batch_size: Prediction batch size
        max_length: Maximum sequence length
    """
    
    print("Generating submission...")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Load and preprocess test data
    print("Loading test data...")
    # The new process writes a single file to the working directory
    test_data_path = os.path.join(working, 'belka.parquet')
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Processed test data not found at {test_data_path}")
    
    df_full = pd.read_parquet(test_data_path)
    df = df_full[df_full['subset'] == 2].copy()  # subset 2 is the test set
    del df_full # free up memory
    
    # Limit to first 1000 samples for testing (remove this for full submission)
    df = df.iloc[:1000].copy()
    
    # Tokenize SMILES
    print("Tokenizing SMILES...")
    df['smiles'] = df['smiles'].apply(atomInSmiles.smiles_tokenizer)
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices({
        'smiles': tf.ragged.constant(df['smiles'].tolist())
    })
    
    # Encode and batch
    encoder = get_smiles_encoder(vocab=vocab, **kwargs)
    ds = ds.map(lambda x: tf.cast(encoder(x['smiles']), dtype=tf.int32))
    ds = ds.padded_batch(batch_size=batch_size, padded_shapes=(max_length,))
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i, batch in enumerate(ds):
        batch_preds = model(batch, training=False)
        predictions.append(batch_preds.numpy())
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_batches} batches")
    
    # Combine predictions
    pred = np.concatenate(predictions, axis=0)
    print(f"Generated predictions shape: {pred.shape}")
    
    # Create submission DataFrame
    protein_cols = ['BRD4_pred', 'HSA_pred', 'sEH_pred']
    df[protein_cols] = pred
    
    # Reshape for submission format
    submission_data = []
    protein_names = ['BRD4', 'HSA', 'sEH']
    
    for i, protein in enumerate(protein_names):
        protein_df = df[[protein, protein_cols[i]]].copy()
        protein_df.columns = ['id', 'binds']
        submission_data.append(protein_df)
    
    # Combine all proteins
    submission = pd.concat(submission_data, axis=0)
    submission = submission.dropna().sort_values(by='id').reset_index(drop=True)
    submission['id'] = submission['id'].astype(int)
    
    # Save submission
    submission_path = os.path.join(working, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission saved to: {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print("\nSubmission preview:")
    print(submission.head())


def predict_molecules(
    model_path: str,
    smiles_list: list,
    vocab: str,
    max_length: int = 128,
    batch_size: int = 32,
    **kwargs) -> np.ndarray:
    """
    Predict binding affinities for a list of SMILES strings.
    
    Args:
        model_path: Path to trained model
        smiles_list: List of SMILES strings
        vocab: Path to vocabulary file
        max_length: Maximum sequence length
        batch_size: Prediction batch size
        
    Returns:
        Array of predictions
    """
    
    print(f"Predicting for {len(smiles_list)} molecules...")
    
    # Load model
    model = load_model(model_path)
    
    # Tokenize SMILES
    tokenized_smiles = [atomInSmiles.smiles_tokenizer(smiles) for smiles in smiles_list]
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices({
        'smiles': tf.ragged.constant(tokenized_smiles)
    })
    
    # Encode and batch
    encoder = get_smiles_encoder(vocab=vocab, **kwargs)
    ds = ds.map(lambda x: tf.cast(encoder(x['smiles']), dtype=tf.int32))
    ds = ds.padded_batch(batch_size=batch_size, padded_shapes=(max_length,))
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Generate predictions
    predictions = []
    for batch in ds:
        batch_preds = model(batch, training=False)
        predictions.append(batch_preds.numpy())
    
    # Combine and return
    pred = np.concatenate(predictions, axis=0)
    return pred


def save_training_config(config: Dict[str, Any], working: str) -> None:
    """
    Save training configuration to a file.
    
    Args:
        config: Configuration dictionary
        working: Working directory
    """
    import json
    
    config_path = os.path.join(working, 'training_config.json')
    
    # Convert non-serializable types
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")


def load_training_config(working: str) -> Dict[str, Any]:
    """
    Load training configuration from a file.
    
    Args:
        working: Working directory
        
    Returns:
        Configuration dictionary
    """
    import json
    
    config_path = os.path.join(working, 'training_config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Training configuration loaded from: {config_path}")
    return config