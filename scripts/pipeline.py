#!/usr/bin/env python3
"""
Belka Transformer Pipeline

Main driver script with CLI interface for the Belka molecular transformer pipeline.
Supports step-by-step execution, configuration management, and comprehensive error handling.
"""

import os
import sys
import argparse
import logging
import traceback
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import psutil

# Add project root to path to enable package imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from src.data_processing import (
        make_parquet,
        get_vocab,
        make_dataset,
        create_validation_dataset,
    )
    from src.training import train_model, evaluate_model, make_submission, save_training_config
    from src.belka_utils import load_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory and that the refactored data_processing.py is in src/")
    sys.exit(1)


def detect_cluster_type() -> str:
    """
    Auto-detect cluster type based on available hardware.
    """
    try:
        # Try to import tensorflow to check for GPU
        import tensorflow as tf
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if gpus:
            logging.info(f"Detected GPU(s): {len(gpus)} device(s)")
            logging.info(f"CPU cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
            return 'gpu'
        else:
            # No GPU available
            if cpu_count >= 8:
                logging.info(f"No GPU detected. CPU cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
                return 'cpu'
            else:
                logging.warning(f"Limited hardware: CPU cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
                return 'unknown'
            
    except ImportError:
        # TensorFlow not available - assume CPU cluster
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logging.warning("TensorFlow not available for GPU detection")
        logging.info(f"CPU cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
        
        if cpu_count >= 8:
            return 'cpu'
        else:
            return 'unknown'
    
    except Exception as e:
        logging.error(f"Error detecting cluster type: {e}")
        return 'unknown'


def get_cluster_config_path(cluster_type: str) -> str:
    """
    Get the configuration file path for the specified cluster type.
    """
    if cluster_type == 'auto':
        cluster_type = detect_cluster_type()
        logging.info(f"Auto-detected cluster type: {cluster_type}")
    
    if cluster_type == 'cpu':
        return 'configs/config_cpu.yaml'
    elif cluster_type == 'gpu':
        return 'configs/config_gpu.yaml'
    else:
        # Unknown or default - use standard config
        logging.warning(f"Unknown cluster type '{cluster_type}', using default config")
        return 'configs/config.yaml'


def validate_cluster_compatibility(config: Dict[str, Any], cluster_type: str) -> None:
    """
    Validate that the current environment is compatible with the cluster configuration.
    """
    compatibility = config.get('compatibility', {})
    
    # Check TensorFlow requirement
    tf_required = compatibility.get('tensorflow_enabled', True)
    if tf_required:
        try:
            import tensorflow as tf
            logging.info(f"TensorFlow version: {tf.__version__}")
        except ImportError:
            if cluster_type == 'gpu':
                raise RuntimeError("TensorFlow required for GPU cluster operations but not available")
            else:
                logging.warning("TensorFlow not available - some operations may fail")
    
    # Check GPU requirement
    gpu_required = compatibility.get('gpu_required', False)
    if gpu_required:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                raise RuntimeError("GPU required but no GPU devices found")
            logging.info(f"GPU validation passed: {len(gpus)} GPU(s) available")
        except ImportError:
            raise RuntimeError("GPU required but TensorFlow not available for GPU detection")
    
    # Check minimum memory
    min_memory = compatibility.get('min_memory_gb', 0)
    available_memory = psutil.virtual_memory().total / (1024**3)
    if available_memory < min_memory:
        logging.warning(f"Available memory ({available_memory:.1f}GB) below recommended minimum ({min_memory}GB)")
    
    # Check minimum CPU cores
    min_cores = compatibility.get('min_cpu_cores', 0)
    available_cores = os.cpu_count() or 1
    if available_cores < min_cores:
        logging.warning(f"Available CPU cores ({available_cores}) below recommended minimum ({min_cores})")
    
    logging.info(f"Cluster compatibility validation passed for {cluster_type} cluster")


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format, handlers=[], force=True)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Configuration loaded from: {config_path}")
    return config


def validate_paths(config: Dict[str, Any]) -> None:
    """
    Validate that required paths exist.
    """
    data_config = config.get('data', {})
    
    root_dir = data_config.get('root')
    if root_dir and not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root data directory not found: {root_dir}")
    
    working_dir = data_config.get('working')
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
        logging.info(f"Working directory: {working_dir}")


def step_generate_parquet(config: Dict[str, Any]) -> None:
    """
    Generate parquet data from raw files using the new robust method.
    """
    logging.info("Step: Generate Parquet Data")
    
    data_config = config.get('data', {})
    system_config = config.get('system', {})
    
    # --- FIX 2: Call the new, unified make_parquet function ---
    logging.info("Using robust, memory-safe parquet generation...")
    # Make a copy to avoid modifying the original config
    system_params = system_config.copy()
    # Pop seed to avoid passing it twice
    seed = system_params.pop('seed', 42)
    make_parquet(
        root=data_config.get('root'),
        working=data_config.get('working'),
        seed=seed,
        **system_params  # Pass other system configs like n_workers
    )
    
    logging.info("Parquet generation completed successfully")


def step_get_vocab(config: Dict[str, Any]) -> None:
    """
    Extract vocabulary from SMILES strings.
    """
    logging.info("Step: Extract Vocabulary")
    data_config = config.get('data', {})
    working = data_config.get('working')
    
    get_vocab(working=working)
    
    vocab_path = os.path.join(working, 'vocab.txt')
    if os.path.exists(vocab_path):
        logging.info(f"Vocabulary file created: {vocab_path}")
    else:
        raise RuntimeError("Failed to create vocabulary file")


def step_make_dataset(config: Dict[str, Any]) -> None:
    """
    Convert parquet to TFRecord dataset.
    """
    logging.info("Step: Create TFRecord Dataset")
    data_config = config.get('data', {})
    working = data_config.get('working')
    
    make_dataset(working=working)
    
    tfr_path = os.path.join(working, 'belka.tfr')
    if os.path.exists(tfr_path):
        logging.info(f"TFRecord dataset created: {tfr_path}")
    else:
        raise RuntimeError("Failed to create TFRecord dataset")


def step_create_val_dataset(config: Dict[str, Any]) -> None:
    """
    Create separate validation dataset.
    """
    logging.info("Step: Create Validation Dataset")
    data_config = config.get('data', {})
    working = data_config.get('working')
    
    create_validation_dataset(working=working)
    
    val_tfr_path = os.path.join(working, 'belka_val.tfr')
    if os.path.exists(val_tfr_path):
        logging.info(f"Validation dataset created: {val_tfr_path}")
    else:
        raise RuntimeError("Failed to create validation dataset")


def step_train_model(config: Dict[str, Any], mode: str, model_path: Optional[str] = None, 
                     initial_epoch: int = 0) -> None:
    """
    Train the transformer model.
    """
    logging.info(f"Step: Train Model (mode: {mode})")
    
    valid_modes = ['mlm', 'fps', 'clf']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    system_config = config.get('system', {})
    
    working = data_config.get('working')
    vocab_path = data_config.get('vocab')
    
    train_params = {
        'mode': mode,
        'working': working,
        'vocab': vocab_path,
        'model_path': model_path,
        'initial_epoch': initial_epoch,
        **model_config,
        **training_config,
        **system_config
    }
    
    save_training_config(train_params, working)
    train_model(**train_params)
    logging.info(f"Model training completed for mode: {mode}")


def run_cpu_preprocess_pipeline(config: Dict[str, Any]) -> None:
    """Runs the dedicated CPU preprocessing steps."""
    logging.info("Running CPU Preprocessing Pipeline (Steps 1-2)")
    try:
        step_generate_parquet(config)
        step_get_vocab(config)
        logging.info("CPU Preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"CPU Preprocessing failed: {e}")
        logging.error(traceback.format_exc())
        raise

def run_gpu_tensorflow_pipeline(config: Dict[str, Any], mode: str) -> None:
    """Runs the dedicated GPU TensorFlow steps."""
    logging.info("Running GPU TensorFlow Pipeline (Steps 3-5)")
    try:
        step_make_dataset(config)
        step_create_val_dataset(config)
        step_train_model(config, mode=mode)
        logging.info("GPU TensorFlow pipeline completed successfully.")
    except Exception as e:
        logging.error(f"GPU pipeline failed: {e}")
        logging.error(traceback.format_exc())
        raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Belka Transformer Pipeline')
    
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--cluster-type', type=str, default='auto', choices=['cpu', 'gpu', 'auto'], help='Cluster type')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    
    # Simplified step selection for clarity
    parser.add_argument('--step', type=str, required=True,
                        choices=['preprocess', 'train'],
                        help="'preprocess' runs data processing on CPU. 'train' runs TF dataset creation and training on GPU.")
    
    parser.add_argument('--mode', type=str, default='clf', choices=['mlm', 'fps', 'clf'], help='Training mode (for train step)')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level, args.log_file)
    
    try:
        cluster_type = args.cluster_type
        if cluster_type == 'auto':
            cluster_type = detect_cluster_type()
            logging.info(f"Auto-detected cluster type: {cluster_type}")
        
        config_path = args.config or get_cluster_config_path(cluster_type)
        config = load_config(config_path)
        
        validate_paths(config)
        
        if args.step == 'preprocess':
            if cluster_type != 'cpu':
                logging.warning(f"Running 'preprocess' step on a non-CPU node ({cluster_type}). This is supported but not optimal.")
            run_cpu_preprocess_pipeline(config)
        
        elif args.step == 'train':
            if cluster_type != 'gpu':
                logging.warning(f"Running 'train' step on a non-GPU node ({cluster_type}). This will likely be very slow or fail.")
            run_gpu_tensorflow_pipeline(config, args.mode)
            
        logging.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
