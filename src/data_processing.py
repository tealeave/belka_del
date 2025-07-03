"""
Belka Data Processing Pipeline - Fully Optimized & Refactored

This module contains functions for data preparation, preprocessing, and dataset creation
for the Belka molecular transformer pipeline. It has been completely refactored for
memory safety and performance on multi-core CPU systems.

Key Principles:
- Dask-centric workflow to avoid loading large data into memory.
- True chunk-to-file processing for raw data reshaping.
- Vectorized operations (NumPy/pandas) instead of slow apply/mapply loops.
- Single, robust 'make_parquet' function, removing redundant and unsafe variants.
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
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from rdkit import Chem
import atomInSmiles
import gc
import time
from typing import Dict, Tuple, Union, Optional, Set

# Local application imports
from .belka_utils import FPGenerator


def monitor_memory(context: str = "") -> None:
    """
    Monitor and print current memory usage.
    """
    memory = psutil.virtual_memory()
    print(
        f"[{context:<25}] Memory Usage: "
        f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB "
        f"({memory.percent:.1f}%)"
    )

def monitor_system_resources(context: str = "") -> None:
    """Enhanced system resource monitoring for HPC environments."""
    # Memory monitoring
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    memory_percent = memory.percent
    
    # CPU monitoring
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Disk I/O monitoring
    disk_io = psutil.disk_io_counters()
    disk_usage = psutil.disk_usage('/')
    disk_used_gb = disk_usage.used / (1024**3)
    disk_total_gb = disk_usage.total / (1024**3)
    disk_percent = (disk_usage.used / disk_usage.total) * 100
    
    logger.info(f"System Resources {context}:")
    logger.info(f"  Memory: {used_gb:.2f}GB/{total_gb:.2f}GB ({memory_percent:.1f}%)")
    logger.info(f"  CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    logger.info(f"  Disk: {disk_used_gb:.2f}GB/{disk_total_gb:.2f}GB ({disk_percent:.1f}%)")
    if disk_io:
        logger.info(f"  Disk I/O: {disk_io.read_bytes/(1024**3):.2f}GB read, {disk_io.write_bytes/(1024**3):.2f}GB written")
    
    # Keep the original print for backward compatibility
    print(f"[{context:<25}] Memory Usage: {used_gb:.1f}GB/{total_gb:.1f}GB ({memory_percent:.1f}%)")

def _canonicalize_smiles_safe(smi: str) -> Optional[str]:
    """Safely canonicalize a single SMILES string, returning None on error."""
    if not isinstance(smi, str):
        return None
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None # Or return smi if you prefer to keep invalid ones

def vectorized_canonical_smiles(smiles_series: pd.Series) -> pd.Series:
    """
    Vectorized replacement of [Dy] DNA linker and subsequent canonicalization.

    Args:
        smiles_series: A pandas Series of SMILES strings.

    Returns:
        A pandas Series of canonicalized SMILES strings.
    """
    # Use fast, vectorized string replacement
    smiles_no_linker = smiles_series.str.replace(r'\[Dy\]', '[H]', regex=True)
    # Use .map for a fast iteration over the series
    return smiles_no_linker.map(_canonicalize_smiles_safe)


def read_and_pivot_parquet_chunks(subset: str, root: str, output_dir: str, **kwargs) -> None:
    """
    Reads a raw Belka parquet file, pivots it in chunks, and writes each
    processed chunk as a separate parquet file to a directory.

    Args:
        subset: Dataset subset ('train' or 'test').
        root: Root directory containing the raw parquet files.
        output_dir: Directory to write the processed chunk files to.
    """
    file_path = os.path.join(root, f'{subset}.parquet')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = kwargs.get('pandas_chunksize', 32000)
    debug_mode = kwargs.get('debug_mode', False)
    debug_max_chunks = kwargs.get('debug_max_chunks', 3)
    
    debug_suffix = f" (DEBUG: max {debug_max_chunks} chunks)" if debug_mode else ""
    print(f"Reading '{subset}.parquet' and writing pivoted chunks to '{output_dir}'{debug_suffix}...")

    parquet_file = pq.ParquetFile(file_path)
    total_chunks = parquet_file.num_row_groups
    chunk_num = 0
    total_rows = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        if len(batch) == 0:
            continue
            
        # DEBUG mode early exit
        if debug_mode and chunk_num >= debug_max_chunks:
            print(f"DEBUG: Stopping after {debug_max_chunks} chunks (processed {total_rows} rows)")
            break

        chunk = batch.to_pandas()
        
        # Rename for consistency
        chunk = chunk.rename(columns={
            'buildingblock1_smiles': 'block1',
            'buildingblock2_smiles': 'block2',
            'buildingblock3_smiles': 'block3',
            'molecule_smiles': 'smiles'})
        
        cols = ['block1', 'block2', 'block3', 'smiles']
        values = 'binds' if subset == 'train' else 'id'
        
        try:
            chunk_pivoted = chunk.pivot(index=cols, columns='protein_name', values=values).reset_index()
        except Exception:
            # Fallback for duplicates or other pivot issues
            chunk_pivoted = chunk.groupby(cols + ['protein_name'])[values].first().unstack('protein_name').reset_index()

        # Fill NaNs that result from pivoting
        protein_cols = ['BRD4', 'HSA', 'sEH']
        for col in protein_cols:
            if col in chunk_pivoted.columns:
                fill_val = 0 if subset == 'train' else -1
                chunk_pivoted[col] = chunk_pivoted[col].fillna(fill_val)

        # Write chunk to its own file
        output_file = os.path.join(output_dir, f'chunk_{chunk_num:04d}.parquet')
        chunk_pivoted.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        total_rows += len(chunk_pivoted)
        chunk_num += 1
        
        # Progress reporting
        if chunk_num % 100 == 0 or debug_mode:
            progress_pct = (chunk_num / total_chunks * 100) if not debug_mode else (chunk_num / min(debug_max_chunks, total_chunks) * 100)
            print(f"  Processed chunk {chunk_num}/{total_chunks if not debug_mode else min(debug_max_chunks, total_chunks)} ({progress_pct:.1f}%) - {total_rows:,} rows")
            monitor_memory(f"After chunk {chunk_num}")
        
        del chunk, chunk_pivoted
        gc.collect()

    debug_suffix = " (DEBUG mode)" if debug_mode else ""
    print(f"✓ Finished pivoting '{subset}.parquet'{debug_suffix}: {total_rows} rows in {chunk_num} chunks.")


def make_parquet(root: str, working: str, seed: int, **kwargs) -> None:
    """
    Creates the final, unified 'belka.parquet' file using a memory-safe,
    Dask-centric, and vectorized workflow. This is the primary function for data processing.
    """
    pipeline_start_time = time.time()
    debug_mode = kwargs.get('debug_mode', False)
    debug_suffix = " (DEBUG MODE)" if debug_mode else ""
    
    logger.info(f"Starting make_parquet pipeline with root='{root}', working='{working}', debug_mode={debug_mode}")
    print(f"Starting make_parquet{debug_suffix}...")
    monitor_system_resources("Start of make_parquet")

    # --- 1. Setup ---
    temp_dir = os.path.join(working, 'temp_processing')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("Reading building blocks for validation split...")
    train_blocks = set()
    try:
        bb_cols = ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles']
        parquet_file = pq.ParquetFile(os.path.join(root, 'train.parquet'))
        total_rows = parquet_file.metadata.num_rows
        processed_rows = 0
        batch_count = 0
        
        logger.info(f"Processing {total_rows:,} rows from train.parquet for building block extraction")
        
        for batch in parquet_file.iter_batches(batch_size=65536, columns=bb_cols):
            batch_count += 1
            chunk = batch.to_pandas()
            processed_rows += len(chunk)
            
            for col in bb_cols:
                train_blocks.update(chunk[col].unique())
            
            if batch_count % 10 == 0:
                progress_pct = (processed_rows / total_rows) * 100
                logger.info(f"Building block extraction: processed {processed_rows:,}/{total_rows:,} rows ({progress_pct:.1f}%) - {len(train_blocks):,} unique blocks found")
                
    except Exception as e:
        raise RuntimeError(f"Could not read building blocks from train.parquet: {e}")

    _, val_blocks = train_test_split(list(train_blocks), test_size=0.03, random_state=seed)
    val_blocks_set = set(val_blocks)
    print(f"✓ Validation blocks selected: {len(val_blocks_set)} blocks")
    del train_blocks, val_blocks
    gc.collect()
    
    # --- 2. Define the core transformation function for a partition ---
    def _transform_partition(df: pd.DataFrame, subset_name: str, val_blocks_set: Optional[Set] = None) -> pd.DataFrame:
        """Applies all vectorized transformations to a Dask partition (a pandas DataFrame)."""
        
        # Extract protein column values BEFORE dropping them
        protein_cols = ['BRD4', 'HSA', 'sEH']
        protein_values = {}
        if subset_name in ['train', 'test']:
            for col in protein_cols:
                if col in df.columns:
                    protein_values[col] = df[col].values
        
        # Drop raw protein columns if they exist
        df = df.drop(columns=[col for col in protein_cols if col in df.columns], errors='ignore')

        # Vectorized 'binds' column creation
        if subset_name == 'train':
            binds_array = np.stack([
                protein_values[col].astype(np.int8) for col in protein_cols
            ], axis=-1)
        elif subset_name == 'test':
            binds_array = np.full((len(df), 3), 2, dtype=np.int8)
        elif subset_name == 'extra':
            binds_array = np.column_stack([
                np.full(len(df), 2, dtype=np.int8),
                np.full(len(df), 2, dtype=np.int8),
                np.clip(df['binds'].values, 0, 1).astype(np.int8)
            ])
        df['binds'] = binds_array.tolist()

        # Vectorized 'smiles_no_linker' creation
        df['smiles_no_linker'] = vectorized_canonical_smiles(df['smiles'])

        # Vectorized 'subset' column creation
        if subset_name == 'train':
            block_cols = ['block1', 'block2', 'block3']
            # Create a boolean mask where True means at least one block is in the validation set
            is_in_val = df[block_cols].isin(val_blocks_set).any(axis=1)
            df['subset'] = is_in_val.astype(np.int8) # 1 for val, 0 for train
        elif subset_name == 'test':
            df['subset'] = np.int8(2)
        else: # extra
            df['subset'] = np.int8(0)

        # Drop building block columns
        df = df.drop(columns=['block1', 'block2', 'block3'], errors='ignore')
        
        # Ensure consistent column order for schema compliance
        column_order = ['smiles', 'binds', 'smiles_no_linker', 'subset']
        return df[column_order]

    # --- 3. Process each data source using Dask ---
    all_ddfs = []
    
    # Process 'train' and 'test' using the same chunked pattern
    for subset_name in ['train', 'test']:
        print(f"Processing '{subset_name}' dataset...")
        chunks_dir = os.path.join(temp_dir, f'{subset_name}_chunks')
        read_and_pivot_parquet_chunks(subset_name, root, chunks_dir, **kwargs)
        
        print(f"  Loading Dask DataFrame for '{subset_name}'...")
        ddf = dd.read_parquet(chunks_dir)
        
        # Define meta for Dask's map_partitions
        meta = {
            'smiles': 'object', 'binds': 'object', 
            'smiles_no_linker': 'object', 'subset': 'int8'
        }
        
        print(f"  Applying transformations to '{subset_name}'...")
        transformed_ddf = ddf.map_partitions(
            _transform_partition, 
            subset_name=subset_name, 
            val_blocks_set=val_blocks_set if subset_name == 'train' else None,
            meta=meta
        )
        all_ddfs.append(transformed_ddf)
        monitor_system_resources(f"Dask graph for '{subset_name}'")

    # Process 'extra' data directly with Dask
    print("Processing 'extra' (DNA) data with Dask...")
    extra_csv_path = os.path.join(root, 'DNA_Labeled_Data.csv')
    extra_ddf = dd.read_csv(extra_csv_path, usecols=['new_structure', 'read_count'], blocksize='64MB')
    extra_ddf = extra_ddf.rename(columns={'new_structure': 'smiles', 'read_count': 'binds'})
    
    meta_extra = {'smiles': 'object', 'binds': 'object', 'smiles_no_linker': 'object', 'subset': 'int8'}
    transformed_extra_ddf = extra_ddf.map_partitions(_transform_partition, subset_name='extra', meta=meta_extra)
    all_ddfs.append(transformed_extra_ddf)
    monitor_system_resources("Dask graph for 'extra'")

    # --- 4. Final Assembly & Save ---
    print("Combining all datasets with Dask...")
    full_ddf = dd.concat(all_ddfs)
    
    print("Shuffling combined data...")
    full_ddf = full_ddf.sample(frac=1.0, random_state=seed).repartition(npartitions=20)
    
    output_path = os.path.join(working, 'belka.parquet')
    # Define proper PyArrow schema with correct column order
    schema = pa.schema([
        ('smiles', pa.string()),
        ('binds', pa.list_(pa.int8(), 3)),
        ('smiles_no_linker', pa.string()),
        ('subset', pa.int8())
    ])

    print(f"Writing final parquet file to: {output_path}")
    monitor_system_resources("Before final to_parquet")
    
    # Verify column order before writing (debugging)
    print(f"DataFrame columns before parquet write: {list(full_ddf.columns)}")
    
    # Use schema to ensure consistent data types and structure
    full_ddf.to_parquet(output_path, engine='pyarrow', compression='snappy', schema=schema)
    monitor_system_resources("After final to_parquet")

    # --- 5. Cleanup ---
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    total_pipeline_time = time.time() - pipeline_start_time
    logger.info(f"make_parquet pipeline completed in {total_pipeline_time:.2f} seconds")
    print(f"\n✓ Parquet creation completed successfully: {output_path}")
    print(f"Total pipeline time: {total_pipeline_time:.2f} seconds")


def get_vocab(working: str, **kwargs) -> None:
    """
    Extract vocabulary from SMILES strings for TextVectorization.
    Optimized with Dask for better memory efficiency.
    """
    parquet_path = os.path.join(working, 'belka.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Cannot find belka.parquet at {parquet_path}. Run make_parquet first.")
        
    print("Reading parquet file with Dask to extract vocabulary...")
    logger.info("Loading parquet file for vocabulary extraction")
    df = dd.read_parquet(parquet_path, columns=['smiles'])
    
    total_partitions = df.npartitions
    logger.info(f"Vocabulary extraction: processing {total_partitions} partitions")

    def get_unique_tokens(partition: pd.DataFrame) -> pd.Series:
        """Tokenizes all SMILES in a partition and returns a single set of unique tokens."""
        all_tokens = set()
        for smi in partition['smiles']:
            all_tokens.update(atomInSmiles.smiles_tokenizer(smi))
        # Return as a list inside a Series for Dask aggregation
        return pd.Series([list(all_tokens)])

    # Apply to each partition
    logger.info("Applying tokenization to all partitions...")
    unique_tokens_per_partition = df.map_partitions(get_unique_tokens, meta=(None, 'object')).compute()
    
    # Combine all unique tokens from all partitions
    logger.info("Combining unique tokens from all partitions...")
    final_vocab = sorted(list(set(itertools.chain.from_iterable(unique_tokens_per_partition.tolist()))))
    
    vocab_path = os.path.join(working, 'vocab.txt')
    logger.info(f"Saving vocabulary to {vocab_path}")
    pd.DataFrame(data=final_vocab).to_csv(vocab_path, index=False, header=False)
    
    logger.info(f"Vocabulary extraction completed: {len(final_vocab)} unique tokens saved")
    print(f"Vocabulary saved to: {vocab_path} ({len(final_vocab)} tokens)")


def make_dataset(working: str, **kwargs) -> None:
    """
    Convert parquet data to TensorFlow dataset format (TFRecord).
    Optimized to process the data in partitions to avoid OOM errors.
    """
    parquet_path = os.path.join(working, 'belka.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Cannot find belka.parquet at {parquet_path}. Run make_parquet first.")

    temp_dir = os.path.join(working, 'temp_tfrecords')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    fp_generator = FPGenerator()
    auto = tf.data.AUTOTUNE

    print("Reading parquet file partitions to create TFRecords...")
    parquet_file = pq.ParquetFile(parquet_path)
    
    for i in range(parquet_file.num_row_groups):
        print(f"Processing partition {i+1}/{parquet_file.num_row_groups}...")
        partition = parquet_file.read_row_group(i).to_pandas()
        
        partition['smiles'] = partition['smiles'].apply(atomInSmiles.smiles_tokenizer)

        def partition_generator():
            for row in partition.itertuples(index=False):
                yield {
                    'smiles': row.smiles,
                    'smiles_no_linker': row.smiles_no_linker,
                    'binds': row.binds,
                    'subset': row.subset
                }

        ds_partition = tf.data.Dataset.from_generator(
            partition_generator,
            output_signature={
                'smiles': tf.TensorSpec(shape=(None,), dtype=tf.string),
                'smiles_no_linker': tf.TensorSpec(shape=(), dtype=tf.string),
                'binds': tf.TensorSpec(shape=(3,), dtype=tf.int8),
                'subset': tf.TensorSpec(shape=(), dtype=tf.int8)
            })
        
        def serialize_and_fingerprint(x):
            x['smiles'] = tf.io.serialize_tensor(x['smiles'])
            x['ecfp'] = fp_generator(x['smiles_no_linker'])
            x.pop('smiles_no_linker')
            return x

        ds_partition = ds_partition.batch(1024).map(serialize_and_fingerprint, num_parallel_calls=auto).unbatch()
        
        partition_path = os.path.join(temp_dir, f'partition_{i:04d}.tfr')
        logger.info(f"Saving TFRecord partition {i+1}/{total_partitions} to {partition_path}")
        ds_partition.save(partition_path, compression='GZIP')
        del partition, ds_partition
        gc.collect()

    print("Combining all partition TFRecord files...")
    partition_files = tf.data.Dataset.list_files(os.path.join(temp_dir, '*.tfr'), shuffle=False)
    
    # Interleave the datasets for performance
    combined_ds = partition_files.interleave(
        lambda x: tf.data.Dataset.load(x, compression='GZIP'),
        cycle_length=auto,
        num_parallel_calls=auto
    )
    
    output_path = os.path.join(working, 'belka.tfr')
    print(f"Saving final TFRecord dataset to: {output_path}")
    combined_ds.save(output_path, compression='GZIP')
    
    print("Cleaning up temporary TFRecord files...")
    shutil.rmtree(temp_dir)
    print("✓ TFRecord dataset creation completed!")


# --- Downstream functions (unchanged as they operate on the generated files) ---

def get_smiles_encoder(vocab: str, **kwargs) -> TextVectorization:
    """
    Create TextVectorization encoder for SMILES strings.
    """
    vocab_path = os.path.join(os.path.dirname(vocab), 'vocab.txt')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please run get_vocab first.")
    
    with open(vocab_path, 'r') as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    tokenizer = TextVectorization(
        standardize=None,
        split=atomInSmiles.smiles_tokenizer,
        vocabulary=vocab_list)
    return tokenizer


def create_validation_dataset(working: str, **kwargs) -> None:
    """
    Create separate validation dataset from the main TFRecord.
    """
    main_tfr_path = os.path.join(working, 'belka.tfr')
    if not os.path.exists(main_tfr_path):
        raise FileNotFoundError(f"Main TFRecord not found at {main_tfr_path}. Run make_dataset first.")
        
    print("Loading main dataset to filter for validation...")
    ds = tf.data.Dataset.load(main_tfr_path, compression='GZIP')
    
    val_ds = ds.filter(lambda x: tf.equal(x['subset'], 1))
    
    val_path = os.path.join(working, 'belka_val.tfr')
    print(f"Saving validation dataset to: {val_path}")
    val_ds.save(val_path, compression='GZIP')
    
    print("✓ Validation dataset created!")


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
    """
    # Get SMILES encoder
    encoder = get_smiles_encoder(vocab=vocab, **kwargs)
    auto = tf.data.AUTOTUNE

    def encode_smiles(x):
        """Encode SMILES strings using TextVectorization."""
        # The tokenizer now handles the splitting
        x['smiles'] = tf.cast(encoder(tf.io.parse_tensor(x['smiles'], out_type=tf.string)), dtype=tf.int32)
        return x

    def get_model_inputs(x):
        """Prepare inputs based on training mode."""
        if mode == 'mlm':
            # MLM masking logic remains the same
            paddings_mask = tf.cast(x['smiles'] != 0, dtype=tf.float32)
            probs = tf.stack([
                1.0 - masking_rate,
                masking_rate * 0.8,
                masking_rate * 0.1,
                masking_rate * 0.1
            ], axis=0)
            probs = tf.expand_dims(probs, axis=0)
            probs = tf.ones_like(x['smiles'], dtype=tf.float32)[:, :4] * probs
            probs = tf.math.log(probs)
            
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

            masked_tokens = tf.multiply(
                mask_decisions,
                tf.stack([
                    x['smiles'],
                    tf.ones_like(x['smiles']),
                    tf.random.uniform(
                        shape=tf.shape(x['smiles']), 
                        minval=2, maxval=vocab_size, dtype=tf.int32),
                    x['smiles']
                ], axis=-1))
            
            x['masked'] = tf.reduce_sum(masked_tokens, axis=-1)
            mask_for_loss = tf.reduce_sum(mask_decisions[:, :, 1:], axis=-1)
            x['smiles'] = tf.stack([
                (x['smiles'] * mask_for_loss) - (1 - mask_for_loss),
                x['smiles']
            ], axis=-1)
            return x['masked'], x['smiles']
        elif mode == 'fps':
            return x['smiles'], x['ecfp']
        else: # clf mode
            return x['smiles'], x['binds']

    # Load datasets
    train_ds = tf.data.Dataset.load(os.path.join(working, 'belka.tfr'), compression='GZIP')
    
    # Define dataset configurations based on mode
    if mode == 'mlm':
        features = ['smiles']
        train_filter = None
        val_ds = None
    elif mode == 'fps':
        features = ['smiles', 'ecfp']
        train_filter = lambda x: tf.not_equal(x['subset'], 1)
        val_ds = tf.data.Dataset.load(os.path.join(working, 'belka_val.tfr'), compression='GZIP')
    else: # clf mode
        features = ['smiles', 'binds']
        train_filter = lambda x: tf.equal(x['subset'], 0)
        val_ds = tf.data.Dataset.load(os.path.join(working, 'belka_val.tfr'), compression='GZIP')

    # Prepare padding shapes
    padded_shapes = {'smiles': (max_length,), 'ecfp': (2048,), 'binds': (3,)}

    # Process training dataset
    if train_filter is not None:
        train_ds = train_ds.filter(train_filter)
    
    train_ds = train_ds.map(lambda x: {key: x[key] for key in features}, num_parallel_calls=auto)
    train_ds = train_ds.cache().repeat().shuffle(buffer_size=buffer_size, seed=seed)
    train_ds = train_ds.padded_batch(batch_size=batch_size, padded_shapes={key: padded_shapes[key] for key in features})
    train_ds = train_ds.map(encode_smiles, num_parallel_calls=auto)
    train_ds = train_ds.map(get_model_inputs, num_parallel_calls=auto)
    train_ds = train_ds.prefetch(auto)

    # Process validation dataset if it exists
    if val_ds is not None:
        val_ds = val_ds.map(lambda x: {key: x[key] for key in features}, num_parallel_calls=auto)
        val_ds = val_ds.cache()
        val_ds = val_ds.padded_batch(batch_size=batch_size, padded_shapes={key: padded_shapes[key] for key in features})
        val_ds = val_ds.map(encode_smiles, num_parallel_calls=auto)
        val_ds = val_ds.map(get_model_inputs, num_parallel_calls=auto)
        val_ds = val_ds.prefetch(auto)

    return train_ds, val_ds