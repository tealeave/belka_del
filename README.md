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
# Quick testing with DEBUG mode (processes only 3 chunks, ~30 seconds)
poetry run python scripts/pipeline.py --step preprocess --debug

# Run the complete pipeline (production)
poetry run python scripts/pipeline.py --step preprocess
poetry run python scripts/pipeline.py --step train --mode clf

# Run individual steps
poetry run python scripts/pipeline.py --step preprocess
poetry run python scripts/pipeline.py --step train --mode clf

# Train different modes
poetry run python scripts/pipeline.py --step train --mode mlm  # Masked Language Model
poetry run python scripts/pipeline.py --step train --mode fps  # Fingerprint Prediction

# Resume training
poetry run python scripts/pipeline.py --step train --mode clf --model-path models/checkpoint.keras --initial-epoch 50
```

### Alternative: Direct Python (if dependencies are installed)

```bash
# If all dependencies are available in your environment
python scripts/pipeline.py --step preprocess
python scripts/pipeline.py --step train --mode clf
```

### Using Python API

```python
from src.data_processing import make_parquet, get_vocab, make_dataset
from src.training import create_model, train_model, evaluate_model
from src.belka_utils import load_model

# Data processing
make_parquet(root='data/raw', working='data/processed', seed=42)
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

- **Data Preparation**: `make_parquet()` - Create unified dataset
- **Vocabulary**: `get_vocab()`, `get_smiles_encoder()` - SMILES tokenization
- **Dataset Creation**: `make_dataset()`, `train_val_datasets()` - TensorFlow datasets

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

### 1. Preprocess Data
Combines train, test, and extra DNA data into a unified format and creates the vocabulary:
```bash
# Quick debug run (3 chunks, ~30 seconds)
poetry run python scripts/pipeline.py --step preprocess --debug

# Full production run (~35+ minutes)
poetry run python scripts/pipeline.py --step preprocess
```

**⚠️ Safety Note**: Preprocessing will fail if output files (`belka.parquet`, `vocab.txt`) already exist. Use `--force-clean` or manual cleanup if needed.

### 2. Train Model
Creates the TFRecord dataset and trains the transformer in the specified mode:
```bash
poetry run python scripts/pipeline.py --step train --mode clf
```

### 3. Generate Submission
Creates a competition submission:
```bash
poetry run python scripts/pipeline.py --step make_submission --model-path models/best_model.keras
```

### Utility Commands
```bash
# Check for existing output files
poetry run python scripts/slurm_utils.py --cluster-type cpu clean --dry-run

# Force cleanup of existing files (use with caution)
poetry run python scripts/slurm_utils.py --cluster-type cpu clean --force-clean

# Create required directories
poetry run python scripts/slurm_utils.py --cluster-type cpu create-dirs

# Validate input files
poetry run python scripts/slurm_utils.py --cluster-type cpu validate-inputs

# Monitor system resources
poetry run python scripts/slurm_utils.py --cluster-type cpu monitor
```

## 🧪 Development & Debugging

### DEBUG Mode
For rapid development and testing, use DEBUG mode which processes only the first 3 chunks of data.

**Run from project root directory:**

```bash
# Quick validation (30 seconds instead of 35+ minutes)
poetry run python scripts/pipeline.py --step preprocess --debug

# SLURM debug job with minimal resources
sbatch slurm/job_cpu_preprocess_debug.sh
```

### Performance Comparison
| Mode | Data Processed | Time | Memory | Use Case |
|------|----------------|------|---------|----------|
| **DEBUG** | ~96K rows (3 chunks) | ~30 seconds | ~4GB | Development, testing fixes |
| **Production** | ~98M rows (all data) | ~35+ minutes | ~128GB | Full preprocessing |

### Development Workflow
1. **Test**: Run with `--debug` to quickly validate code changes
2. **Validate**: Check output files and logs for correctness  
3. **Deploy**: Run full pipeline for production data

### When to Use DEBUG Mode
- ✅ Testing code changes and bug fixes
- ✅ Validating new features  
- ✅ Quick pipeline verification
- ✅ Learning the codebase
- ❌ Final production runs
- ❌ Full dataset analysis

### SLURM Integration
**Important**: Run these commands from the project root directory (`/pub/ddlin/projects/belka_del`)

```bash
# Quick debug testing (minimal resources)
sbatch slurm/job_cpu_preprocess_debug.sh

# Full production run (full resources)  
sbatch slurm/job_cpu_preprocess.sh

# Alternative: Environment variable debug mode
DEBUG_MODE=true sbatch slurm/job_cpu_preprocess.sh
```

**⚠️ Safety Note**: SLURM jobs will fail if output files (`belka.parquet`, `vocab.txt`) already exist. This prevents accidental data loss. See [Error Handling](#-error-handling) section for details.

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
poetry run python scripts/pipeline.py --config custom_config.yaml --step train
```

### Override Parameters
```bash
poetry run python scripts/pipeline.py --step train --batch-size 1024 --epochs 500
```

### Logging and Monitoring
```bash
poetry run python scripts/pipeline.py --step train --log-level DEBUG --log-file training.log
```

### SLURM Cluster Usage
For cluster environments, use the provided SLURM scripts. **Run from project root directory:**

```bash
# Quick debug testing (3 chunks, minimal resources)
sbatch slurm/job_cpu_preprocess_debug.sh

# Full production preprocessing (all data, full resources)
sbatch slurm/job_cpu_preprocess.sh

# Alternative: Use environment variable for debug mode
DEBUG_MODE=true sbatch slurm/job_cpu_preprocess.sh

# GPU training (after preprocessing)
sbatch slurm/job_gpu_training.sh
```

**SLURM Resource Allocation:**
- **Debug mode**: 8 CPUs, 64GB RAM, 10-minute limit
- **Production mode**: 16 CPUs, 128GB RAM, no time limit

**⚠️ Important**: SLURM jobs now include safety checks and will fail if output files already exist. This prevents accidental data loss in batch processing environments.

## 🚨 Troubleshooting

### Common Issues

#### SLURM Job Failures Due to Existing Files
```bash
# ❌ ERROR: Job fails with "Cleanup failed - existing output files detected"
sbatch slurm/job_cpu_preprocess.sh

# ✅ SOLUTION 1: Check what files exist
poetry run python scripts/slurm_utils.py --cluster-type cpu clean --dry-run

# ✅ SOLUTION 2: Force cleanup (CAUTION: deletes existing files)
poetry run python scripts/slurm_utils.py --cluster-type cpu clean --force-clean

# ✅ SOLUTION 3: Manual cleanup
rm -f data/raw/belka.parquet data/raw/vocab.txt
sbatch slurm/job_cpu_preprocess.sh
```

#### SLURM Script Path Errors
```bash
# ❌ ERROR: Unable to open file job_cpu_preprocess.sh
sbatch job_cpu_preprocess.sh

# ✅ CORRECT: Include the slurm/ directory path
sbatch slurm/job_cpu_preprocess.sh

# ✅ ALTERNATIVE: Change to slurm directory first
cd slurm && sbatch job_cpu_preprocess.sh && cd ..
```

#### Working Directory Context
All commands in this README assume you're in the project root directory:
```bash
# Verify you're in the correct directory
pwd
# Should show: /pub/ddlin/projects/belka_del (or your installation path)

# If not, navigate to project root
cd /path/to/your/belka_del
```

#### Poetry Environment Issues
```bash
# ❌ If you get "command not found" errors
python scripts/pipeline.py --step preprocess

# ✅ Always use poetry run for consistent environment
poetry run python scripts/pipeline.py --step preprocess

# Verify Poetry environment is active
poetry env info
```

#### DEBUG Mode Not Working
```bash
# ❌ Wrong: Debug flag without slurm/ path
DEBUG_MODE=true sbatch job_cpu_preprocess.sh

# ✅ Correct: Include full path
DEBUG_MODE=true sbatch slurm/job_cpu_preprocess.sh

# ✅ Alternative: Use dedicated debug script
sbatch slurm/job_cpu_preprocess_debug.sh
```

#### File Not Found Errors
Ensure required data files exist:
```bash
# Check for required input files
ls -la data/raw/train.parquet
ls -la data/raw/test.parquet  
ls -la data/raw/DNA_Labeled_Data.csv
```

## 🔄 Migration from Legacy Code

The original notebook code is preserved in `src/belka.py` with compatibility functions:

```python
# Legacy usage (still supported)
from src.belka import set_parameters, train_val_set, make_parquet

parameters = set_parameters(
    root='/path/to/data',
    working='/path/to/working',
    mode='clf',
    # ... other parameters
)

# New modular usage (recommended)
from src.data_processing import make_parquet
from src.training import train_model

make_parquet(root='/path/to/data', working='/path/to/working', seed=42)
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
- **⚠️ Safety Checks**: Prevents accidental deletion of existing output files

### Safety Check for Existing Files

The pipeline now includes safety checks to prevent accidental deletion of processed data:

```bash
# ❌ This will ERROR OUT if belka.parquet or vocab.txt already exist
poetry run python scripts/slurm_utils.py --cluster-type cpu clean

# ✅ Preview what would be deleted without actually deleting
poetry run python scripts/slurm_utils.py --cluster-type cpu clean --dry-run

# ⚠️ Force deletion of existing files (use with caution)
poetry run python scripts/slurm_utils.py --cluster-type cpu clean --force-clean
```

**Error Message Example:**
```
✗ Cannot proceed: Existing output files detected!
Found files that would be overwritten:
  • belka.parquet (1.2GB, modified: Mon Jul  3 15:30:45 2023)
  • vocab.txt (2.1KB, modified: Mon Jul  3 15:28:12 2023)
Use --force-clean flag to overwrite existing files.
```

### SLURM Job Behavior

SLURM jobs will now fail cleanly with clear error messages if output files exist:

```bash
# These jobs will fail if output files exist
sbatch slurm/job_cpu_preprocess.sh
sbatch slurm/job_cpu_preprocess_debug.sh

# Check SLURM logs for error details
cat logs/cpu_preprocess_*.err
```

**Options when job fails:**
1. **Manual cleanup**: Delete/backup files manually, then rerun
2. **Force mode**: Modify scripts to use `--force-clean` flag
3. **Dry run**: Check what would be deleted first

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
