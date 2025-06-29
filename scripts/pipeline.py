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

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    from data_processing import (
        make_parquet, make_parquet_memory_safe, get_vocab, make_dataset,
        create_validation_dataset, initialize_mapply
    )
    from training import train_model, evaluate_model, make_submission, save_training_config
    from belka_utils import load_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def detect_cluster_type() -> str:
    """
    Auto-detect cluster type based on available hardware.
    
    Returns:
        Cluster type: 'cpu', 'gpu', or 'unknown'
    """
    try:
        # Try to import tensorflow to check for GPU
        import tensorflow as tf
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if gpus:
            # GPU available - likely GPU cluster
            logging.info(f"Detected GPU(s): {len(gpus)} device(s)")
            logging.info(f"CPU cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
            
            if cpu_count <= 4:
                # Low CPU count + GPU = GPU cluster
                return 'gpu'
            else:
                # High CPU count + GPU = workstation/dev environment
                return 'gpu'  # Still prefer GPU cluster config
        else:
            # No GPU available
            if cpu_count >= 8:
                # High CPU count, no GPU = CPU cluster
                logging.info(f"No GPU detected. CPU cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
                return 'cpu'
            else:
                # Low CPU count, no GPU = limited environment
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
    
    Args:
        cluster_type: Cluster type ('cpu', 'gpu', or 'auto')
        
    Returns:
        Path to appropriate configuration file
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
    
    Args:
        config: Configuration dictionary
        cluster_type: Detected or specified cluster type
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
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
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
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
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
    
    Args:
        config: Configuration dictionary
    """
    data_config = config.get('data', {})
    
    # Check root directory
    root_dir = data_config.get('root')
    if root_dir and not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root data directory not found: {root_dir}")
    
    # Check working directory and create if needed
    working_dir = data_config.get('working')
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
        logging.info(f"Working directory: {working_dir}")


def step_generate_parquet(config: Dict[str, Any], memory_safe: bool = True) -> None:
    """
    Generate parquet data from raw files.
    
    Args:
        config: Configuration dictionary
        memory_safe: Use memory-safe processing
    """
    logging.info("Step: Generate Parquet Data")
    
    data_config = config.get('data', {})
    system_config = config.get('system', {})
    
    # Initialize parallel processing
    n_workers = system_config.get('n_workers', -1)
    initialize_mapply(n_workers=n_workers)
    
    # Choose parquet generation method
    if memory_safe:
        logging.info("Using memory-safe parquet generation")
        make_parquet_memory_safe(
            root=data_config.get('root'),
            working=data_config.get('working'),
            seed=system_config.get('seed', 42))
    else:
        logging.info("Using standard parquet generation")
        make_parquet(
            root=data_config.get('root'),
            working=data_config.get('working'),
            seed=system_config.get('seed', 42))
    
    logging.info("Parquet generation completed successfully")


def step_get_vocab(config: Dict[str, Any]) -> None:
    """
    Extract vocabulary from SMILES strings.
    
    Args:
        config: Configuration dictionary
    """
    logging.info("Step: Extract Vocabulary")
    
    data_config = config.get('data', {})
    working = data_config.get('working')
    
    # Check if parquet file exists
    parquet_path = os.path.join(working, 'belka.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    get_vocab(working=working)
    
    # Verify vocab file was created
    vocab_path = os.path.join(working, 'vocab.txt')
    if os.path.exists(vocab_path):
        logging.info(f"Vocabulary file created: {vocab_path}")
    else:
        raise RuntimeError("Failed to create vocabulary file")


def step_make_dataset(config: Dict[str, Any]) -> None:
    """
    Convert parquet to TFRecord dataset.
    
    Args:
        config: Configuration dictionary
    """
    logging.info("Step: Create TFRecord Dataset")
    
    data_config = config.get('data', {})
    working = data_config.get('working')
    
    # Check if parquet file exists
    parquet_path = os.path.join(working, 'belka.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    make_dataset(working=working)
    
    # Verify TFRecord was created
    tfr_path = os.path.join(working, 'belka.tfr')
    if os.path.exists(tfr_path):
        logging.info(f"TFRecord dataset created: {tfr_path}")
    else:
        raise RuntimeError("Failed to create TFRecord dataset")


def step_create_val_dataset(config: Dict[str, Any]) -> None:
    """
    Create separate validation dataset.
    
    Args:
        config: Configuration dictionary
    """
    logging.info("Step: Create Validation Dataset")
    
    data_config = config.get('data', {})
    working = data_config.get('working')
    
    # Check if main TFRecord exists
    tfr_path = os.path.join(working, 'belka.tfr')
    if not os.path.exists(tfr_path):
        raise FileNotFoundError(f"Main TFRecord not found: {tfr_path}")
    
    create_validation_dataset(working=working)
    
    # Verify validation dataset was created
    val_tfr_path = os.path.join(working, 'belka_val.tfr')
    if os.path.exists(val_tfr_path):
        logging.info(f"Validation dataset created: {val_tfr_path}")
    else:
        raise RuntimeError("Failed to create validation dataset")


def step_train_model(config: Dict[str, Any], mode: str, model_path: Optional[str] = None, 
                     initial_epoch: int = 0) -> None:
    """
    Train the transformer model.
    
    Args:
        config: Configuration dictionary
        mode: Training mode ('mlm', 'fps', or 'clf')
        model_path: Path to existing model (for resuming)
        initial_epoch: Starting epoch for resumed training
    """
    logging.info(f"Step: Train Model (mode: {mode})")
    
    # Validate mode
    valid_modes = ['mlm', 'fps', 'clf']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    # Get configuration sections
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    system_config = config.get('system', {})
    
    # Check required files
    working = data_config.get('working')
    vocab_path = data_config.get('vocab')
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    # Prepare training parameters
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
    
    # Save training configuration
    save_training_config(train_params, working)
    
    # Train model
    model = train_model(**train_params)
    
    logging.info(f"Model training completed for mode: {mode}")
    return model


def step_make_submission(config: Dict[str, Any], model_path: str) -> None:
    """
    Generate competition submission.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
    """
    logging.info("Step: Generate Submission")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Generate submission
    make_submission(
        model_path=model_path,
        working=data_config.get('working'),
        vocab=data_config.get('vocab'),
        root=data_config.get('root'),
        batch_size=training_config.get('batch_size', 1024),
        max_length=config.get('model', {}).get('max_length', 128))
    
    logging.info("Submission generated successfully")


def step_evaluate_model(config: Dict[str, Any], model_path: str, mode: str) -> None:
    """
    Evaluate a trained model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
        mode: Model mode ('mlm', 'fps', or 'clf')
    """
    logging.info("Step: Evaluate Model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Get configuration
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        working=data_config.get('working'),
        vocab=data_config.get('vocab'),
        mode=mode,
        batch_size=training_config.get('batch_size', 1024),
        max_length=model_config.get('max_length', 128))
    
    logging.info("Model evaluation completed")


def run_full_pipeline(config: Dict[str, Any], mode: str = 'clf', memory_safe: bool = True) -> None:
    """
    Run the complete pipeline from data processing to training.
    
    Args:
        config: Configuration dictionary
        mode: Training mode
        memory_safe: Use memory-safe processing
    """
    logging.info("Running full pipeline")
    
    try:
        # Step 1: Generate parquet data
        step_generate_parquet(config, memory_safe=memory_safe)
        
        # Step 2: Extract vocabulary
        step_get_vocab(config)
        
        # Step 3: Create TFRecord dataset
        step_make_dataset(config)
        
        # Step 4: Create validation dataset
        step_create_val_dataset(config)
        
        # Step 5: Train model
        step_train_model(config, mode=mode)
        
        logging.info("Full pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        logging.error(traceback.format_exc())
        raise


def step_preprocess(config: Dict[str, Any], cluster_type: str) -> None:
    """
    Run CPU cluster preprocessing steps (Steps 1-2).
    
    Args:
        config: Configuration dictionary
        cluster_type: Cluster type
    """
    logging.info("Step: Preprocess (CPU Cluster - Steps 1-2)")
    
    if cluster_type != 'cpu':
        logging.warning(f"Running preprocess steps on {cluster_type} cluster - consider using CPU cluster")
    
    # Check if preprocessing is enabled
    pipeline_config = config.get('pipeline', {})
    enabled_steps = pipeline_config.get('enabled_steps', [])
    
    # Step 1: Generate Parquet Data
    if 'generate_parquet' in enabled_steps:
        step_generate_parquet(config, memory_safe=True)
    else:
        logging.info("Step 1 (generate_parquet) disabled for this cluster type")
    
    # Step 2: Extract Vocabulary
    if 'get_vocab' in enabled_steps:
        step_get_vocab(config)
    else:
        logging.info("Step 2 (get_vocab) disabled for this cluster type")
    
    logging.info("Preprocessing completed - ready for transfer to GPU cluster")


def step_tensorflow_pipeline(config: Dict[str, Any], cluster_type: str, mode: str = 'clf') -> None:
    """
    Run GPU cluster TensorFlow pipeline (Steps 3-6).
    
    Args:
        config: Configuration dictionary
        cluster_type: Cluster type
        mode: Training mode
    """
    logging.info("Step: TensorFlow Pipeline (GPU Cluster - Steps 3-6)")
    
    if cluster_type != 'gpu':
        logging.warning(f"Running TensorFlow pipeline on {cluster_type} cluster - consider using GPU cluster")
    
    # Check if TensorFlow operations are enabled
    pipeline_config = config.get('pipeline', {})
    enabled_steps = pipeline_config.get('enabled_steps', [])
    
    # Step 3: Create TFRecord Dataset
    if 'make_dataset' in enabled_steps:
        step_make_dataset(config)
    else:
        logging.info("Step 3 (make_dataset) disabled for this cluster type")
    
    # Step 4: Create Validation Dataset
    if 'create_val_dataset' in enabled_steps:
        step_create_val_dataset(config)
    else:
        logging.info("Step 4 (create_val_dataset) disabled for this cluster type")
    
    # Step 5: Train Model
    if 'train_model' in enabled_steps:
        step_train_model(config, mode)
    else:
        logging.info("Step 5 (train_model) disabled for this cluster type")
    
    logging.info("TensorFlow pipeline completed")


def validate_step_compatibility(step: str, cluster_type: str, config: Dict[str, Any]) -> bool:
    """
    Validate that a step is compatible with the current cluster type.
    
    Args:
        step: Step name
        cluster_type: Cluster type
        config: Configuration dictionary
        
    Returns:
        True if compatible, False otherwise
    """
    pipeline_config = config.get('pipeline', {})
    enabled_steps = pipeline_config.get('enabled_steps', [])
    disabled_steps = pipeline_config.get('disabled_steps', [])
    
    # Check if step is explicitly disabled
    if step in disabled_steps:
        logging.error(f"Step '{step}' is disabled for {cluster_type} cluster")
        return False
    
    # Check if step is in enabled list (if specified)
    if enabled_steps and step not in enabled_steps:
        logging.error(f"Step '{step}' is not enabled for {cluster_type} cluster")
        logging.info(f"Enabled steps for {cluster_type}: {enabled_steps}")
        return False
    
    # Step-specific compatibility checks
    cpu_only_steps = ['generate_parquet', 'get_vocab']
    gpu_preferred_steps = ['make_dataset', 'create_val_dataset', 'train_model', 'make_submission']
    
    if step in cpu_only_steps and cluster_type == 'gpu':
        logging.warning(f"Step '{step}' is typically run on CPU cluster but can work on GPU cluster")
    
    if step in gpu_preferred_steps and cluster_type == 'cpu':
        logging.warning(f"Step '{step}' requires TensorFlow and is typically run on GPU cluster")
        # Check if TensorFlow is available
        if not config.get('compatibility', {}).get('tensorflow_enabled', False):
            logging.error(f"Step '{step}' requires TensorFlow but it's disabled for CPU cluster")
            return False
    
    return True


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        'model': {
            'depth': 32,
            'num_heads': 8,
            'num_layers': 4,
            'dropout_rate': 0.1,
            'activation': 'gelu',
            'max_length': 128,
            'vocab_size': 43
        },
        'training': {
            'batch_size': 2048,
            'buffer_size': 10000000,
            'epochs': 1000,
            'patience': 20,
            'steps_per_epoch': 10000,
            'validation_steps': 2000,
            'masking_rate': 0.15,
            'epsilon': 1e-07
        },
        'data': {
            'root': '/pub/ddlin/projects/belka_del/data/raw',
            'working': '/pub/ddlin/projects/belka_del/data/raw',
            'vocab': '/pub/ddlin/projects/belka_del/data/raw/vocab.txt'
        },
        'system': {
            'seed': 42,
            'n_workers': 8
        }
    }


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Belka Transformer Pipeline')
    
    # General arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (auto-selected if not specified)')
    parser.add_argument('--cluster-type', type=str, default='auto',
                        choices=['cpu', 'gpu', 'auto'],
                        help='Cluster type (auto-detect if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    # Pipeline step selection
    parser.add_argument('--step', type=str, default='full',
                        choices=['full', 'preprocess', 'tensorflow_pipeline',
                                'generate_parquet', 'get_vocab', 'make_dataset',
                                'create_val_dataset', 'train_model', 'evaluate_model', 'make_submission'],
                        help='Pipeline step to execute')
    
    # Step-specific arguments
    parser.add_argument('--mode', type=str, default='clf',
                        choices=['mlm', 'fps', 'clf'],
                        help='Training mode')
    parser.add_argument('--memory-safe', action='store_true', default=True,
                        help='Use memory-safe data processing')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='Starting epoch for resumed training')
    
    # Configuration overrides
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--working-dir', type=str, default=None,
                        help='Override working directory')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Determine cluster type
        cluster_type = args.cluster_type
        if cluster_type == 'auto':
            cluster_type = detect_cluster_type()
            logging.info(f"Auto-detected cluster type: {cluster_type}")
        
        # Load cluster-specific configuration
        if args.config:
            config_path = args.config
        else:
            config_path = get_cluster_config_path(cluster_type)
            logging.info(f"Using cluster-specific config: {config_path}")
        
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            logging.warning(f"Config file not found: {config_path}")
            logging.info("Using default configuration")
            config = create_default_config()
        
        # Validate cluster compatibility
        validate_cluster_compatibility(config, cluster_type)
        
        # Apply command line overrides
        if args.batch_size:
            config.setdefault('training', {})['batch_size'] = args.batch_size
        if args.epochs:
            config.setdefault('training', {})['epochs'] = args.epochs
        if args.working_dir:
            config.setdefault('data', {})['working'] = args.working_dir
        
        # Validate paths
        validate_paths(config)
        
        # Execute requested step with cluster-aware routing
        if args.step == 'full':
            run_full_pipeline(config, mode=args.mode, memory_safe=args.memory_safe)
        elif args.step == 'preprocess':
            step_preprocess(config, cluster_type)
        elif args.step == 'tensorflow_pipeline':
            step_tensorflow_pipeline(config, cluster_type, args.mode)
        elif args.step == 'generate_parquet':
            if validate_step_compatibility(args.step, cluster_type, config):
                step_generate_parquet(config, memory_safe=args.memory_safe)
        elif args.step == 'get_vocab':
            if validate_step_compatibility(args.step, cluster_type, config):
                step_get_vocab(config)
        elif args.step == 'make_dataset':
            if validate_step_compatibility(args.step, cluster_type, config):
                step_make_dataset(config)
        elif args.step == 'create_val_dataset':
            if validate_step_compatibility(args.step, cluster_type, config):
                step_create_val_dataset(config)
        elif args.step == 'train_model':
            if validate_step_compatibility(args.step, cluster_type, config):
                step_train_model(config, args.mode, args.model_path, args.initial_epoch)
        elif args.step == 'evaluate_model':
            if not args.model_path:
                raise ValueError("--model-path required for evaluation")
            if validate_step_compatibility('train_model', cluster_type, config):  # Same requirements as training
                step_evaluate_model(config, args.model_path, args.mode)
        elif args.step == 'make_submission':
            if not args.model_path:
                raise ValueError("--model-path required for submission")
            if validate_step_compatibility(args.step, cluster_type, config):
                step_make_submission(config, args.model_path)
        
        logging.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()