# Belka Transformer - Modular Implementation

A production-ready molecular transformer pipeline for binding affinity prediction, restructured into modular components for maintainability and scalability.

## 🏗️ Architecture Overview

The codebase has been restructured from a monolithic notebook implementation into a modular, production-ready pipeline:

```
belka_del/
├── src/                      # Core modules
│   ├── __init__.py          # Package initialization
│   ├── belka_utils.py       # Custom layers, losses, metrics, model
│   ├── data_processing.py   # Data pipeline functions
│   ├── training.py          # Training and evaluation logic
│   └── belka.py            # Legacy compatibility module
├── scripts/                 # Executable scripts
│   └── pipeline.py         # Main CLI pipeline script
├── configs/                 # Configuration files
│   └── config.yaml         # Default configuration
├── data/                   # Data directory
│   └── raw/               # Raw data files
├── models/                 # Saved models
└── notebooks/             # Original notebooks (preserved)
```

## 🚀 Quick Start

### Using Poetry (Recommended)

This project uses Poetry for dependency management. All commands should be run with `poetry run`:

```bash
# Run the complete pipeline
poetry run python scripts/pipeline.py --step full --mode clf

# Run individual steps
poetry run python scripts/pipeline.py --step generate_parquet --memory-safe
poetry run python scripts/pipeline.py --step get_vocab
poetry run python scripts/pipeline.py --step make_dataset
poetry run python scripts/pipeline.py --step train_model --mode clf
poetry run python scripts/pipeline.py --step make_submission --model-path models/best_model.keras

# Train different modes
poetry run python scripts/pipeline.py --step train_model --mode mlm  # Masked Language Model
poetry run python scripts/pipeline.py --step train_model --mode fps  # Fingerprint Prediction
poetry run python scripts/pipeline.py --step train_model --mode clf  # Classification

# Resume training
poetry run python scripts/pipeline.py --step train_model --mode clf --model-path models/checkpoint.keras --initial-epoch 50
```

### Alternative: Direct Python (if dependencies are installed)

```bash
# If all dependencies are available in your environment
python scripts/pipeline.py --step full --mode clf
```

### Using Python API

```python
from src import train_model, make_parquet_memory_safe, get_vocab, make_dataset
from src.training import create_model, evaluate_model
from src.belka_utils import Belka, load_model

# Data processing
make_parquet_memory_safe(root='data/raw', working='data/processed', seed=42)
get_vocab(working='data/processed')
make_dataset(working='data/processed')

# Model training
model = create_model(mode='clf', vocab_size=43)
trained_model = train_model(
    mode='clf',
    model_name='belka_clf',
    working='data/processed',
    vocab='data/processed/vocab.txt')

# Evaluation
results = evaluate_model(trained_model, working='data/processed', mode='clf')
```

## 📁 Module Details

### `src/belka_utils.py`
Core components for the transformer architecture:

- **Custom Losses**: `MultiLabelLoss`, `CategoricalLoss`, `BinaryLoss`
- **Custom Metrics**: `MaskedAUC` 
- **Custom Layers**: `Encodings`, `Embeddings`, `FeedForward`, `SelfAttention`, `EncoderLayer`
- **Model**: `Belka` transformer model
- **Utilities**: `FPGenerator`, `load_model`

### `src/data_processing.py`
Data pipeline functions:

- **Data Reading**: `read_parquet()` - Read and preprocess train/test data
- **Data Preparation**: `make_parquet()`, `make_parquet_memory_safe()` - Create unified dataset
- **Vocabulary**: `get_vocab()`, `get_smiles_encoder()` - SMILES tokenization
- **Dataset Creation**: `make_dataset()`, `train_val_datasets()` - TensorFlow datasets
- **Utilities**: `initialize_mapply()` for parallel processing

### `src/training.py`
Training and evaluation functions:

- **Model Management**: `create_model()`, `setup_callbacks()`
- **Training**: `train_model()`, `evaluate_model()`
- **Inference**: `make_submission()`, `predict_molecules()`
- **Configuration**: `save_training_config()`, `load_training_config()`

### `scripts/pipeline.py`
CLI interface with comprehensive options:

- **Step Execution**: Individual or full pipeline execution
- **Configuration Management**: YAML-based configuration
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging system

## ⚙️ Configuration

Configuration is managed through `configs/config.yaml`:

```yaml
model:
  depth: 32                    # Hidden dimension
  num_heads: 8                 # Attention heads
  num_layers: 4                # Transformer layers
  dropout_rate: 0.1            # Dropout rate
  max_length: 128              # Sequence length
  vocab_size: 43               # Vocabulary size

training:
  batch_size: 2048             # Batch size
  epochs: 1000                 # Max epochs
  patience: 20                 # Early stopping patience
  steps_per_epoch: 10000       # Steps per epoch

data:
  root: "/path/to/raw/data"    # Raw data directory
  working: "/path/to/working"  # Working directory
  vocab: "/path/to/vocab.txt"  # Vocabulary file
```

## 🔧 Pipeline Steps

### 1. Generate Parquet Data
Combines train, test, and extra DNA data into a unified format:
```bash
poetry run python scripts/pipeline.py --step generate_parquet --memory-safe
```

### 2. Extract Vocabulary
Creates vocabulary from SMILES tokens:
```bash
poetry run python scripts/pipeline.py --step get_vocab
```

### 3. Create TFRecord Dataset
Converts data to optimized TensorFlow format:
```bash
poetry run python scripts/pipeline.py --step make_dataset
```

### 4. Create Validation Dataset
Separates validation subset:
```bash
poetry run python scripts/pipeline.py --step create_val_dataset
```

### 5. Train Model
Trains transformer in specified mode:
```bash
poetry run python scripts/pipeline.py --step train_model --mode clf
```

### 6. Generate Submission
Creates competition submission:
```bash
poetry run python scripts/pipeline.py --step make_submission --model-path models/best_model.keras
```

## 🎯 Training Modes

### MLM (Masked Language Model)
Pretraining on SMILES strings with masked token prediction:
- **Purpose**: Learn molecular representations
- **Input**: SMILES with random masking
- **Output**: Predicted masked tokens
- **Use Case**: Pretraining for downstream tasks

### FPS (Fingerprint Prediction) 
Predict molecular fingerprints from SMILES:
- **Purpose**: Learn fingerprint representations
- **Input**: SMILES strings
- **Output**: ECFP fingerprints (2048-dim)
- **Use Case**: Molecular similarity and clustering

### CLF (Classification)
Binding affinity prediction for three proteins:
- **Purpose**: Predict binding to BRD4, HSA, sEH
- **Input**: SMILES strings
- **Output**: Binding probabilities (3 classes)
- **Use Case**: Drug discovery and screening

## 🛠️ Advanced Usage

### Custom Configuration
```bash
poetry run python scripts/pipeline.py --config custom_config.yaml --step train_model
```

### Override Parameters
```bash
poetry run python scripts/pipeline.py --step train_model --batch-size 1024 --epochs 500
```

### Memory-Safe Processing
For large datasets:
```bash
poetry run python scripts/pipeline.py --step generate_parquet --memory-safe
```

### Logging and Monitoring
```bash
poetry run python scripts/pipeline.py --step full --log-level DEBUG --log-file training.log
```

## 🔄 Migration from Legacy Code

The original notebook code is preserved in `src/belka.py` with compatibility functions:

```python
# Legacy usage (still supported)
from src.belka import set_parameters, train_val_set, make_parquet_memory_safe

parameters = set_parameters(
    root='/path/to/data',
    working='/path/to/working',
    mode='clf',
    # ... other parameters
)

# New modular usage (recommended)
from src.data_processing import make_parquet_memory_safe
from src.training import train_model

make_parquet_memory_safe(root='/path/to/data', working='/path/to/working', seed=42)
model = train_model(mode='clf', working='/path/to/working', ...)
```

## 🚀 Performance Optimizations

### Memory Management
- Memory-safe data processing for large datasets
- Chunked processing to prevent OOM errors
- Efficient TensorFlow dataset pipeline

### Training Optimizations
- Mixed precision training support
- Gradient accumulation for large effective batch sizes
- Dynamic learning rate scheduling
- Early stopping with best model restoration

### Data Pipeline
- Parallel data processing with configurable workers
- Dataset caching for faster iterations
- Optimized TFRecord format for training

## 🔧 Error Handling

The pipeline includes comprehensive error handling:

- **File Validation**: Checks for required input files
- **Memory Management**: Automatic fallback to memory-safe processing
- **Checkpoint Recovery**: Resume training from saved checkpoints
- **GPU Fallback**: Automatic CPU fallback if GPU fails
- **Detailed Logging**: Comprehensive logging for debugging

## 📊 Monitoring and Logging

### Training Metrics
- Loss curves for all training modes
- Validation metrics (when applicable)
- Learning rate scheduling
- Model checkpointing

### System Monitoring
- Memory usage tracking
- GPU utilization (if available)
- Processing time for each step
- Data pipeline performance

## 🧪 Testing

Test individual components:

```python
# Test model creation
from src.training import create_model
model = create_model(mode='clf', vocab_size=43)

# Test data processing
from src.data_processing import get_vocab
get_vocab(working='data/processed')

# Test prediction
from src.training import predict_molecules
predictions = predict_molecules(
    model_path='models/model.keras',
    smiles_list=['CCO', 'C1CCCCC1'],
    vocab='data/vocab.txt')
```

## 📚 Dependencies

Core dependencies:
- TensorFlow 2.x
- pandas, numpy
- scikit-learn
- RDKit
- scikit-fingerprints (skfp)
- dask, pyarrow
- atomInSmiles
- mapply, einops
- PyYAML

## 🤝 Contributing

When contributing to this modular implementation:

1. **Follow the modular structure**: Keep functionality separated by module
2. **Update configuration**: Add new parameters to `config.yaml`
3. **Add CLI support**: Extend `scripts/pipeline.py` for new features
4. **Maintain backward compatibility**: Update `src/belka.py` if needed
5. **Add comprehensive logging**: Use the logging system for debugging
6. **Write tests**: Test individual components and integration

## 📖 Documentation

- **Module Documentation**: Each module has comprehensive docstrings
- **Configuration Reference**: `configs/config.yaml` is fully documented
- **CLI Help**: Run `python scripts/pipeline.py --help` for all options
- **Legacy Reference**: `src/pipeline_plan.md` contains the original plan

This modular implementation provides a robust, scalable foundation for molecular transformer research and production deployments.