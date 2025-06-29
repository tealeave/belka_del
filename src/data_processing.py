"""
Belka Data Processing Pipeline

This module contains functions for data preparation, preprocessing, and dataset creation
for the Belka molecular transformer pipeline.
"""

import os
import shutil
import itertools
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import dask.dataframe as dd
import pyarrow as pa
from sklearn.model_selection import train_test_split
from rdkit import Chem
import atomInSmiles
import mapply
from typing import Dict, Tuple, Union

from .belka_utils import FPGenerator


def monitor_memory() -> Dict[str, float]:
    """
    Monitor current memory usage.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent
    }


def read_parquet(subset: str, root: str, **kwargs) -> pd.DataFrame:
    """
    Read and preprocess train/test parquet files with memory-safe chunked processing.
    
    Args:
        subset: Dataset subset ('train' or 'test')
        root: Root directory containing the parquet files
        
    Returns:
        Preprocessed DataFrame
    """
    import gc
    
    file_path = os.path.join(root, f'{subset}.parquet')
    
    # Initial memory check
    mem_info = monitor_memory()
    print(f"Memory at start: {mem_info['used_gb']:.1f}GB used, {mem_info['available_gb']:.1f}GB available ({mem_info['percent_used']:.1f}%)")
    
    # Get chunk size from config, default to 2000
    chunk_size = kwargs.get('pandas_chunksize', 2000)
    
    # Adaptive chunk sizing based on available memory
    if mem_info['available_gb'] < 32:  # Less than 32GB available
        chunk_size = min(chunk_size, 1000)
        print(f"⚠️  Low memory detected, reducing chunk size to {chunk_size}")
    elif mem_info['available_gb'] < 16:  # Less than 16GB available
        chunk_size = min(chunk_size, 500)
        print(f"⚠️  Very low memory detected, reducing chunk size to {chunk_size}")
    
    print(f"Reading {subset}.parquet in chunks of {chunk_size}...")
    
    try:
        # Use pyarrow for batch reading
        parquet_file = pa.parquet.ParquetFile(file_path)
        processed_chunks = []
        
        # Read in batches for memory efficiency
        batch_reader = parquet_file.iter_batches(batch_size=chunk_size)
        
        chunk_num = 0
        for batch in batch_reader:
            chunk_num += 1
            
            # Memory check before processing each chunk
            mem_info = monitor_memory()
            print(f"Processing chunk {chunk_num} ({len(batch)} rows) - Memory: {mem_info['used_gb']:.1f}GB/{mem_info['total_gb']:.1f}GB ({mem_info['percent_used']:.1f}%)")
            
            # Memory safety check
            if mem_info['percent_used'] > 90:
                print(f"⚠️  Memory usage high ({mem_info['percent_used']:.1f}%), forcing garbage collection")
                gc.collect()
                mem_info = monitor_memory()
                if mem_info['percent_used'] > 95:
                    raise MemoryError(f"Memory usage too high ({mem_info['percent_used']:.1f}%), aborting to prevent OOM")
            
            # Convert batch to pandas DataFrame
            chunk = batch.to_pandas()
            
            if len(chunk) == 0:
                continue
                
            # Rename columns for consistency
            chunk = chunk.rename(columns={
                'buildingblock1_smiles': 'block1',
                'buildingblock2_smiles': 'block2', 
                'buildingblock3_smiles': 'block3',
                'molecule_smiles': 'smiles'})
            
            # Group by molecule -> get multiclass labels
            cols = ['block1', 'block2', 'block3', 'smiles']
            values = 'binds' if subset == 'train' else 'id'
            
            try:
                chunk_pivoted = chunk.pivot(index=cols, columns='protein_name', values=values).reset_index()
                processed_chunks.append(chunk_pivoted)
            except Exception as e:
                print(f"Warning: Could not pivot chunk {chunk_num}, trying alternative approach: {e}")
                try:
                    # Alternative: group and aggregate instead of pivot
                    chunk_grouped = chunk.groupby(cols + ['protein_name'])[values].first().unstack('protein_name').reset_index()
                    processed_chunks.append(chunk_grouped)
                except Exception as e2:
                    print(f"Error: Both pivot and unstack failed for chunk {chunk_num}: {e2}")
                    # Fallback: save raw chunk and continue
                    processed_chunks.append(chunk)
            
            # Force garbage collection between chunks
            del chunk
            gc.collect()
        
        print("Combining chunks...")
        mem_info = monitor_memory()
        print(f"Memory before combining: {mem_info['used_gb']:.1f}GB used ({mem_info['percent_used']:.1f}%)")
        
        # Concatenate all processed chunks
        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
            
            # Fill any NaN values that might have been created during pivot/unstack
            protein_cols = ['BRD4', 'HSA', 'sEH']
            for col in protein_cols:
                if col in df.columns:
                    if subset == 'train':
                        df[col] = df[col].fillna(0)  # Fill missing binds with 0
                    else:
                        df[col] = df[col].fillna(-1)  # Fill missing IDs with -1
        else:
            # Empty result
            df = pd.DataFrame()
        
        # Final cleanup
        del processed_chunks
        gc.collect()
        
        # Final memory check
        mem_info = monitor_memory()
        print(f"Completed reading {subset}.parquet: {len(df)} rows")
        print(f"Final memory usage: {mem_info['used_gb']:.1f}GB used ({mem_info['percent_used']:.1f}%)")
        
        return df
        
    except MemoryError as e:
        print(f"❌ Memory error during processing: {e}")
        # Attempt emergency cleanup
        gc.collect()
        raise
    except Exception as e:
        print(f"❌ Error reading {subset}.parquet: {e}")
        # Attempt emergency cleanup
        gc.collect()
        raise


def make_parquet(root: str, working: str, seed: int, **kwargs) -> None:
    """
    Create unified parquet dataset from train, test, and extra data.
    
    This function combines all data sources, creates validation splits,
    processes SMILES strings, and saves the result as a parquet file.
    
    Args:
        root: Root directory containing raw data files
        working: Working directory for output
        seed: Random seed for reproducibility
    """
    
    def validation_split(x, test_blocks: set):
        """
        Determine train (0) or validation (1) subset based on building block overlap.
        """
        blocks = set(x[col] for col in ['block1', 'block2', 'block3'])
        overlap = len(blocks.intersection(test_blocks))
        return np.int8(0 if overlap == 0 else 1)

    def replace_linker(smiles: str) -> str:
        """
        Replace [Dy] DNA linker with hydrogen atom.
        """
        smiles = smiles.replace('[Dy]', '[H]')
        return Chem.CanonSmiles(smiles)

    # Process each subset
    dataset = []
    for subset in ['test', 'extra', 'train']:
        
        print(f"Processing '{subset}' subset...")
        
        # Read data
        if subset in ['train', 'test']:
            df = read_parquet(subset=subset, root=root, **kwargs)
        else:
            # Read extra DNA data
            df = pd.read_csv(
                os.path.join(root, 'DNA_Labeled_Data.csv'), 
                usecols=['new_structure', 'read_count'])
            df = df.rename(columns={'new_structure': 'smiles', 'read_count': 'binds'})

        # Stack binding affinity labels [BRD4, HSA, sEH]
        protein_cols = ['BRD4', 'HSA', 'sEH']
        if subset == 'train':
            # Stack actual binding labels
            df['binds'] = np.stack(
                [df[col].to_numpy() for col in protein_cols], 
                axis=-1, dtype=np.int8).tolist()
        elif subset == 'test':
            # Set placeholder labels (2 = missing/unknown)
            df["binds"] = np.tile(
                np.array([[2, 2, 2]], dtype=np.int8), 
                reps=(df.shape[0], 1)).tolist()
        else:
            # Extra data: only sEH binding, others missing
            df['binds'] = df['binds'].mapply(
                lambda x: [2, 2, np.clip(x, a_min=0, a_max=1)])
        
        # Remove individual protein columns
        for col in protein_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Create validation split for training data
        if subset == 'train':
            # Get all unique building blocks
            all_blocks = (set(df['block1'].tolist()) | 
                         set(df['block2'].tolist()) | 
                         set(df['block3'].tolist()))
            
            # Split blocks for validation (3% of blocks)
            _, val_blocks, _, _ = train_test_split(
                list(all_blocks), list(all_blocks), 
                test_size=0.03, random_state=seed)
            
            # Assign subset labels based on block overlap
            df['subset'] = df.mapply(
                lambda x: validation_split(x, set(val_blocks)), axis=1)
        elif subset == 'test':
            df['subset'] = 2  # Test subset
        else:
            df['subset'] = 0  # Extra data for training only

        # Remove building block columns (no longer needed)
        block_cols = ['block1', 'block2', 'block3']
        for col in block_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Replace [Dy] DNA linker with [H]
        df['smiles_no_linker'] = df['smiles'].mapply(replace_linker)
        
        dataset.append(df)

    # Combine all subsets
    print("Combining and shuffling data...")
    df = pd.concat(dataset, ignore_index=True)
    df = df.sample(frac=1.0, ignore_index=True, random_state=seed)
    
    # Convert to Dask DataFrame for efficient processing
    df = dd.from_pandas(df, npartitions=20)

    # Save to parquet with schema specification
    print("Saving parquet file...")
    output_path = os.path.join(working, 'belka.parquet')
    df.to_parquet(output_path, schema={
        'smiles': pa.string(),
        'binds': pa.list_(pa.int8(), 3),
        'subset': pa.int8(),
        'smiles_no_linker': pa.string()})
    
    print(f"Parquet file saved to: {output_path}")


def make_parquet_memory_safe(root: str, working: str, seed: int, **kwargs) -> None:
    """
    Memory-efficient version of make_parquet that processes data in chunks
    and uses temporary files to avoid memory overflow.
    
    Args:
        root: Root directory containing raw data files
        working: Working directory for output
        seed: Random seed for reproducibility
    """
    
    def validation_split(x, test_blocks: set):
        """
        Determine train (0) or validation (1) subset based on building block overlap.
        """
        blocks = set(x[col] for col in ['block1', 'block2', 'block3'])
        overlap = len(blocks.intersection(test_blocks))
        return np.int8(0 if overlap == 0 else 1)

    def replace_linker(smiles: str) -> str:
        """
        Replace [Dy] DNA linker with hydrogen atom.
        """
        smiles = smiles.replace('[Dy]', '[H]')
        return Chem.CanonSmiles(smiles)

    # Create temporary directory
    temp_dir = os.path.join(working, 'temp_parquet')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print("Processing subsets and saving to temporary files...")
    
    # Process each subset and save to temporary files
    for i, subset in enumerate(['test', 'extra', 'train']):
        print(f"Processing '{subset}' subset...")
        
        # Read data
        if subset in ['train', 'test']:
            df = read_parquet(subset=subset, root=root, **kwargs)
        else:
            df = pd.read_csv(
                os.path.join(root, 'DNA_Labeled_Data.csv'), 
                usecols=['new_structure', 'read_count'])
            df = df.rename(columns={'new_structure': 'smiles', 'read_count': 'binds'})

        # Stack binding affinity labels
        protein_cols = ['BRD4', 'HSA', 'sEH']
        if subset == 'train':
            df['binds'] = np.stack(
                [df[col].to_numpy() for col in protein_cols], 
                axis=-1, dtype=np.int8).tolist()
        elif subset == 'test':
            df["binds"] = np.tile(
                np.array([[2, 2, 2]], dtype=np.int8), 
                reps=(df.shape[0], 1)).tolist()
        else:
            # Process extra data in chunks to avoid memory issues
            df['binds'] = df['binds'].apply(
                lambda x: [2, 2, np.clip(x, a_min=0, a_max=1)])
        
        # Remove protein columns
        for col in protein_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Create validation split
        if subset == 'train':
            all_blocks = (set(df['block1'].tolist()) | 
                         set(df['block2'].tolist()) | 
                         set(df['block3'].tolist()))
            _, val_blocks, _, _ = train_test_split(
                list(all_blocks), list(all_blocks), 
                test_size=0.03, random_state=seed)
            df['subset'] = df.mapply(
                lambda x: validation_split(x, set(val_blocks)), axis=1)
        elif subset == 'test':
            df['subset'] = 2
        else:
            df['subset'] = 0

        # Remove building block columns
        block_cols = ['block1', 'block2', 'block3']
        for col in block_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Replace DNA linker
        df['smiles_no_linker'] = df['smiles'].mapply(replace_linker)

        # Save to temporary file
        temp_file = os.path.join(temp_dir, f'subset_{i}.parquet')
        df.to_parquet(temp_file)
        print(f"Saved temporary file: {temp_file}")

    # Read all temporary files with Dask and combine
    print("Reading temporary files with Dask...")
    df = dd.read_parquet(temp_dir)
    
    # Shuffle and repartition
    print("Shuffling data...")
    df = df.sample(frac=1.0, random_state=seed)
    df = df.repartition(npartitions=20)

    # Save final parquet file
    print("Writing final parquet file...")
    output_path = os.path.join(working, 'belka.parquet')
    df.to_parquet(output_path, schema={
        'smiles': pa.string(),
        'binds': pa.list_(pa.int8(), 3),
        'subset': pa.int8(),
        'smiles_no_linker': pa.string()})
        
    # Clean up temporary files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print(f"Memory-safe parquet creation completed: {output_path}")


def get_vocab(working: str, **kwargs) -> None:
    """
    Extract vocabulary from SMILES strings for TextVectorization.
    
    Args:
        working: Working directory containing belka.parquet
    """
    print("Reading parquet file...")
    df = dd.read_parquet(os.path.join(working, 'belka.parquet'))
    df = df.compute()

    print("Tokenizing SMILES and extracting vocabulary...")
    # Tokenize SMILES and get unique tokens per molecule
    df['smiles'] = df['smiles'].mapply(
        lambda x: list(set(atomInSmiles.smiles_tokenizer(x))))
    
    # Get all unique tokens across all molecules
    vocab = np.unique(
        list(itertools.chain.from_iterable(df['smiles'].tolist()))).tolist()
    
    # Save vocabulary
    vocab_path = os.path.join(working, 'vocab.txt')
    vocab_df = pd.DataFrame(data=vocab)
    vocab_df.to_csv(vocab_path, index=False, header=False)
    
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Vocabulary size: {len(vocab)} tokens")


def get_smiles_encoder(vocab: str, **kwargs) -> TextVectorization:
    """
    Create TextVectorization encoder for SMILES strings.
    
    Args:
        vocab: Path to vocabulary file
        
    Returns:
        Configured TextVectorization layer
    """
    tokenizer = TextVectorization(
        standardize=None,
        split=None,
        vocabulary=vocab)
    return tokenizer


def make_dataset(working: str, **kwargs) -> None:
    """
    Convert parquet data to TensorFlow dataset format (TFRecord).
    
    This function reads the processed parquet file, tokenizes SMILES strings,
    generates molecular fingerprints, and saves everything as a TFRecord dataset.
    
    Args:
        working: Working directory containing belka.parquet
    """
    
    def data_generator():
        """Generator function for creating TensorFlow dataset."""
        for row in df.itertuples(index=False, name='Row'):
            yield {
                'smiles': row.smiles,
                'smiles_no_linker': row.smiles_no_linker,
                'binds': row.binds,
                'subset': row.subset}

    def serialize_smiles(x):
        """Serialize SMILES tokens to tensor format."""
        x['smiles'] = tf.io.serialize_tensor(x['smiles'])
        return x

    def get_ecfp(x):
        """Generate ECFP fingerprints from SMILES."""
        x['ecfp'] = fp_generator(x['smiles_no_linker'])
        x.pop('smiles_no_linker')
        return x

    print("Reading parquet file...")
    df = dd.read_parquet(os.path.join(working, 'belka.parquet'))
    df = df.compute()

    print("Tokenizing SMILES...")
    # Tokenize SMILES strings
    df['smiles'] = df['smiles'].mapply(atomInSmiles.smiles_tokenizer)

    print("Creating TensorFlow dataset...")
    # Create fingerprint generator
    fp_generator = FPGenerator()
    
    # Create TensorFlow dataset from generator
    auto = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_generator(
        generator=data_generator,
        output_signature={
            'smiles': tf.TensorSpec(shape=(None,), dtype=tf.string),
            'smiles_no_linker': tf.TensorSpec(shape=(), dtype=tf.string),
            'binds': tf.TensorSpec(shape=(3,), dtype=tf.int8),
            'subset': tf.TensorSpec(shape=(), dtype=tf.int8)})

    # Process dataset: serialize SMILES -> batch -> generate fingerprints -> unbatch
    ds = ds.map(serialize_smiles, num_parallel_calls=auto)
    ds = ds.batch(batch_size=1024, num_parallel_calls=auto)
    ds = ds.map(get_ecfp, num_parallel_calls=auto)
    ds = ds.unbatch()

    # Save dataset
    output_path = os.path.join(working, 'belka.tfr')
    print(f"Saving TFRecord dataset to: {output_path}")
    ds.save(output_path, compression='GZIP')
    
    print("TFRecord dataset creation completed!")


def create_validation_dataset(working: str, **kwargs) -> None:
    """
    Create separate validation dataset from the main TFRecord.
    
    Args:
        working: Working directory containing belka.tfr
    """
    print("Loading main dataset...")
    ds = tf.data.Dataset.load(os.path.join(working, 'belka.tfr'), compression='GZIP')
    
    print("Filtering validation subset...")
    # Filter for validation subset (subset == 1)  
    val_ds = ds.filter(lambda x: tf.equal(x['subset'], 1))
    
    # Save validation dataset
    val_path = os.path.join(working, 'belka_val.tfr')
    print(f"Saving validation dataset to: {val_path}")
    val_ds.save(val_path, compression='GZIP')
    
    print("Validation dataset created!")


def train_val_datasets(
    batch_size: int, 
    buffer_size: int, 
    masking_rate: float, 
    max_length: int, 
    mode: str, 
    seed: int,
    vocab_size: int, 
    working: str, 
    vocab: str,
    **kwargs) -> Tuple[tf.data.Dataset, Union[tf.data.Dataset, None]]:
    """
    Create training and validation datasets for model training.
    
    Args:
        batch_size: Batch size for training
        buffer_size: Shuffle buffer size
        masking_rate: Rate for MLM masking (only used in MLM mode)
        max_length: Maximum sequence length for padding
        mode: Training mode ('mlm', 'fps', or 'clf')
        seed: Random seed
        vocab_size: Vocabulary size
        working: Working directory
        vocab: Path to vocabulary file
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    
    # Get SMILES encoder
    encoder = get_smiles_encoder(vocab=vocab, **kwargs)
    auto = tf.data.AUTOTUNE

    def encode_smiles(x):
        """Encode SMILES strings using TextVectorization."""
        x['smiles'] = tf.io.parse_tensor(x['smiles'], out_type=tf.string)
        x['smiles'] = tf.cast(encoder(x['smiles']), dtype=tf.int32)
        return x

    def get_model_inputs(x):
        """
        Prepare inputs based on training mode.
        
        MLM mode: Apply random masking to tokens
        FPS mode: Return SMILES and fingerprints  
        CLF mode: Return SMILES and binding labels
        """
        if mode == 'mlm':
            # MLM masking strategy:
            # 80% of time: replace with [MASK] token (1)
            # 10% of time: replace with random token
            # 10% of time: keep original token
            
            # Get padding mask (non-zero tokens)
            paddings_mask = tf.cast(x['smiles'] != 0, dtype=tf.float32)

            # Create masking probabilities
            probs = tf.stack([
                1.0 - masking_rate,        # Keep original
                masking_rate * 0.8,        # Replace with [MASK]
                masking_rate * 0.1,        # Replace with random
                masking_rate * 0.1         # Keep original (but still count as masked)
            ], axis=0)
            
            probs = tf.expand_dims(probs, axis=0)
            probs = tf.ones_like(x['smiles'], dtype=tf.float32)[:, :4] * probs
            probs = tf.math.log(probs)
            
            # Sample masking decisions
            mask_decisions = tf.multiply(
                tf.one_hot(
                    indices=tf.random.categorical(
                        logits=probs, 
                        num_samples=tf.shape(x['smiles'])[-1]),
                    depth=4,
                    dtype=tf.float32,
                    seed=seed),
                tf.expand_dims(paddings_mask, axis=-1))
            mask_decisions = tf.cast(mask_decisions, dtype=tf.int32)

            # Apply masking decisions
            masked_tokens = tf.multiply(
                mask_decisions,
                tf.stack([
                    x['smiles'],  # Keep original
                    tf.ones_like(x['smiles']),  # [MASK] token
                    tf.random.uniform(  # Random token
                        shape=tf.shape(x['smiles']), 
                        minval=2, maxval=vocab_size, dtype=tf.int32),
                    x['smiles']   # Keep original
                ], axis=-1))
            
            x['masked'] = tf.reduce_sum(masked_tokens, axis=-1)
            
            # Create mask for loss computation (1 for masked tokens)
            mask_for_loss = tf.reduce_sum(mask_decisions[:, :, 1:], axis=-1)
            
            # Set non-masked tokens to -1 for loss masking
            x['smiles'] = tf.stack([
                (x['smiles'] * mask_for_loss) - (1 - mask_for_loss),
                x['smiles']
            ], axis=-1)

            return x['masked'], x['smiles']

        elif mode == 'fps':
            return x['smiles'], x['ecfp']
        else:  # clf mode
            return x['smiles'], x['binds']

    # Load datasets
    train_ds = tf.data.Dataset.load(os.path.join(working, 'belka.tfr'), compression='GZIP')
    
    # Define dataset configurations based on mode
    if mode == 'mlm':
        features = ['smiles']
        train_filter = None  # Use all data for MLM
        val_ds = None
    elif mode == 'fps':
        features = ['smiles', 'ecfp']
        train_filter = lambda x: tf.not_equal(x['subset'], 1)  # Exclude validation
        val_ds = tf.data.Dataset.load(os.path.join(working, 'belka_val.tfr'), compression='GZIP')
    else:  # clf mode
        features = ['smiles', 'binds']
        train_filter = lambda x: tf.equal(x['subset'], 0)  # Only training subset
        val_ds = tf.data.Dataset.load(os.path.join(working, 'belka_val.tfr'), compression='GZIP')

    # Prepare padding shapes
    padded_shapes = {
        'smiles': (max_length,), 
        'ecfp': (2048,), 
        'binds': (3,)
    }

    # Process training dataset
    if train_filter is not None:
        train_ds = train_ds.filter(train_filter)
    
    # Select relevant features
    train_ds = train_ds.map(
        lambda x: {key: x[key] for key in features}, 
        num_parallel_calls=auto)
    
    # Cache, repeat, shuffle for training
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat().shuffle(buffer_size=buffer_size, seed=seed)
    
    # Encode SMILES and batch
    train_ds = train_ds.map(encode_smiles, num_parallel_calls=auto)
    train_ds = train_ds.padded_batch(
        batch_size=batch_size, 
        padded_shapes={key: padded_shapes[key] for key in features})
    
    # Get model inputs
    train_ds = train_ds.map(get_model_inputs, num_parallel_calls=auto)
    train_ds = train_ds.prefetch(auto)

    # Process validation dataset if it exists
    if val_ds is not None:
        val_ds = val_ds.map(
            lambda x: {key: x[key] for key in features}, 
            num_parallel_calls=auto)
        val_ds = val_ds.cache()
        val_ds = val_ds.map(encode_smiles, num_parallel_calls=auto)
        val_ds = val_ds.padded_batch(
            batch_size=batch_size,
            padded_shapes={key: padded_shapes[key] for key in features})
        val_ds = val_ds.map(get_model_inputs, num_parallel_calls=auto)
        val_ds = val_ds.prefetch(auto)

    return train_ds, val_ds


def initialize_mapply(n_workers: int = -1, **kwargs) -> None:
    """
    Initialize mapply for parallel processing.
    
    Args:
        n_workers: Number of workers (-1 for all available cores)
    """
    mapply.init(n_workers=n_workers, progressbar=True)
    print(f"Mapply initialized with {n_workers} workers")