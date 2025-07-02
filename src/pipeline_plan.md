# Belka Transformer Pipeline Plan

## Overview
This pipeline implements a molecular transformer for the Belka competition, handling SMILES molecular representation learning and binding affinity prediction. The pipeline supports three modes: MLM (Masked Language Model), FPS (Fingerprint prediction), and CLF (Classification).

## Project Structure
```
/pub/ddlin/projects/belka_del/
├── data/
│   └── raw/
│       ├── DNA_Labeled_Data.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       ├── test.parquet
│       ├── train.csv
│       └── train.parquet
├── scripts/
├── src/
├── configs/
├── models/
└── notebooks/
```

## Proposed Script Structure

### 1. `src/belka_utils.py`
**Purpose**: Core utility functions and custom layers/losses
**Contents**:
- Custom Keras layers (Encodings, Embeddings, FeedForward, SelfAttention, EncoderLayer)
- Custom losses (MultiLabelLoss, CategoricalLoss, BinaryLoss)
- Custom metrics (MaskedAUC)
- Model definition (Belka transformer)
- Data processing utilities (FPGenerator)

### 2. `src/data_processing.py`
**Purpose**: Data preparation and preprocessing functions
**Contents**:
- `read_parquet()` - Read and preprocess train/test parquet files
- `make_parquet()` - Original memory-intensive version
- `make_parquet_memory_safe()` - Memory-efficient version using temporary files
- `make_dataset()` - Convert parquet to TFRecord format
- `get_vocab()` - Extract vocabulary from SMILES tokens
- `get_smiles_encoder()` - Create TextVectorization encoder

### 3. `src/training.py`
**Purpose**: Model training and evaluation functions
**Contents**:
- `train_val_set()` - Create train/validation datasets
- `train_model()` - Main training function with callbacks
- `load_model()` - Model loading with custom objects
- `make_submission()` - Generate competition submission

### 4. `scripts/pipeline.py`
**Purpose**: Main driver script with CLI interface
**Contents**:
- Argument parsing for different pipeline steps
- Configuration management
- Logging setup
- Error handling and recovery
- Step-by-step execution options

### 5. `configs/config.yaml`
**Purpose**: Configuration file for hyperparameters and paths
**Contents**:
- Model hyperparameters
- Data paths
- Training parameters
- Logging configuration

## Pipeline Steps

### Step 1: Generate Parquet Data
**Command**: `python scripts/pipeline.py --step generate_parquet`
**Function**: `make_parquet_memory_safe()` (recommended) or `make_parquet()`
**Input**: 
- `/pub/ddlin/projects/belka_del/data/raw/train.parquet`
- `/pub/ddlin/projects/belka_del/data/raw/test.parquet`
- `/pub/ddlin/projects/belka_del/data/raw/DNA_Labeled_Data.csv`
**Output**: `/pub/ddlin/projects/belka_del/data/raw/belka.parquet`
**Description**: 
- Combines train, test, and extra DNA data
- Processes SMILES strings and binding affinity labels
- Creates validation split based on building block overlap
- Replaces [Dy] DNA linkers with [H]

### Step 2: Generate Vocabulary
**Command**: `python scripts/pipeline.py --step get_vocab`
**Function**: `get_vocab()`
**Input**: `/pub/ddlin/projects/belka_del/data/raw/belka.parquet`
**Output**: `/pub/ddlin/projects/belka_del/data/raw/vocab.txt`
**Description**: 
- Tokenizes all SMILES strings
- Extracts unique tokens
- Saves vocabulary for TextVectorization

### Step 3: Create TFRecord Dataset
**Command**: `python scripts/pipeline.py --step make_dataset`
**Function**: `make_dataset()`
**Input**: `/pub/ddlin/projects/belka_del/data/raw/belka.parquet`
**Output**: `/pub/ddlin/projects/belka_del/data/raw/belka.tfr`
**Description**: 
- Converts parquet to TensorFlow dataset format
- Generates ECFP fingerprints
- Serializes SMILES tokens
- Optimizes for training performance

### Step 4: Create Validation Dataset
**Command**: `python scripts/pipeline.py --step create_val_dataset`
**Function**: Custom extraction from main dataset
**Input**: `/pub/ddlin/projects/belka_del/data/raw/belka.tfr`
**Output**: `/pub/ddlin/projects/belka_del/data/raw/belka_val.tfr`
**Description**: 
- Separates validation subset (subset == 1)
- Creates dedicated validation TFRecord

### Step 5: Train Model
**Command**: `python scripts/pipeline.py --step train_model --mode {clf|fps|mlm}`
**Function**: `train_model()`
**Input**: 
- `/pub/ddlin/projects/belka_del/data/raw/belka.tfr`
- `/pub/ddlin/projects/belka_del/data/raw/belka_val.tfr`
- `/pub/ddlin/projects/belka_del/data/raw/vocab.txt`
**Output**: Model checkpoints in working directory
**Description**: 
- Supports three training modes:
  - `clf`: Classification for binding affinity prediction
  - `fps`: Fingerprint prediction
  - `mlm`: Masked language model pretraining

### Step 6: Generate Submission
**Command**: `python scripts/pipeline.py --step make_submission --model_path {path_to_model}`
**Function**: `make_submission()`
**Input**: 
- Trained model
- Test dataset
**Output**: `/pub/ddlin/projects/belka_del/data/raw/submission.csv`

## Configuration Parameters

### Model Architecture
```yaml
model:
  depth: 32              # Hidden dimension
  num_heads: 8           # Multi-head attention heads
  num_layers: 4          # Transformer encoder layers
  dropout_rate: 0.1      # Dropout rate
  activation: "gelu"     # Activation function
  max_length: 128        # Maximum sequence length
  vocab_size: 43         # Vocabulary size (N+2 for PAD, MASK)

training:
  batch_size: 2048       # Training batch size
  buffer_size: 10000000  # Shuffle buffer size
  epochs: 1000           # Maximum epochs
  patience: 20           # Early stopping patience
  steps_per_epoch: 10000 # Steps per epoch
  validation_steps: 2000 # Validation steps
  masking_rate: 0.15     # MLM masking rate
  epsilon: 1e-07         # Numerical stability
  
data:
  root: "/pub/ddlin/projects/belka_del/data/raw"
  working: "/pub/ddlin/projects/belka_del/data/raw"
  vocab: "/pub/ddlin/projects/belka_del/data/raw/vocab.txt"
  
system:
  seed: 42
  n_workers: 8           # Parallel processing workers
```

## Error Handling & Logging

### Logging Strategy
- **INFO**: Step completion, progress updates
- **DEBUG**: Detailed processing information
- **WARNING**: Non-fatal issues (missing files, fallback methods)
- **ERROR**: Fatal errors with recovery suggestions
- **File logging**: Timestamped logs for each pipeline run

### Error Recovery
- **Memory errors**: Automatic fallback to memory-safe alternatives
- **GPU errors**: CPU fallback options
- **File not found**: Clear instructions for required files
- **Checkpoint recovery**: Resume training from last saved model

### Validation Checks
- Input file existence and format validation
- Memory usage monitoring
- Data shape and type verification
- Model architecture compatibility checks

## CLI Interface

### Basic Usage
```bash
# Run full pipeline
python scripts/pipeline.py --config configs/config.yaml

# Run individual steps
python scripts/pipeline.py --step generate_parquet --memory_safe
python scripts/pipeline.py --step get_vocab
python scripts/pipeline.py --step make_dataset
python scripts/pipeline.py --step train_model --mode clf

# Resume training
python scripts/pipeline.py --step train_model --mode clf --resume --model_path models/checkpoint.keras

# Custom configuration
python scripts/pipeline.py --step train_model --mode clf --config custom_config.yaml --batch_size 1024
```

### Advanced Options
```bash
# Memory management
--memory_safe              # Use memory-efficient processing
--batch_size INT           # Override batch size
--n_workers INT            # Number of parallel workers

# Training options
--resume                   # Resume from checkpoint
--model_path PATH          # Specific model path
--initial_epoch INT        # Starting epoch for resume

# Output control
--output_dir PATH          # Custom output directory
--log_level {DEBUG,INFO,WARNING,ERROR}
--quiet                    # Suppress progress bars
```

## Dependencies
- TensorFlow 2.x
- pandas
- numpy
- dask
- pyarrow
- scikit-learn
- scikit-fingerprints (skfp)
- RDKit
- mapply
- einops
- atomInSmiles
- PyYAML (for config)
- argparse (built-in)

## Success Criteria
1. **Step Independence**: Each step can run independently with proper inputs
2. **Error Recovery**: Clear error messages with actionable solutions
3. **Memory Efficiency**: Handles large datasets without memory overflow
4. **Reproducibility**: Consistent results across runs with same seed
5. **Monitoring**: Comprehensive logging for debugging and progress tracking
6. **Flexibility**: Easy parameter adjustment via CLI or config files

## Next Steps
1. Review and approve this plan
2. Implement `src/belka_utils.py` with all custom components
3. Implement `src/data_processing.py` with data pipeline functions
4. Implement `src/training.py` with training logic
5. Implement `scripts/pipeline.py` with CLI interface
6. Create `configs/config.yaml` with default parameters
7. Test each pipeline step individually
8. Test full pipeline end-to-end