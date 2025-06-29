# Belka Transformer - HPC Cluster Usage Guide

This guide explains how to run the Belka transformer pipeline on HPC environments with separate CPU and GPU clusters.

## Architecture Overview

The pipeline is designed for dual-cluster HPC environments:

- **CPU Cluster**: High CPU cores (16+), no GPU access
  - Runs data preprocessing (Steps 1-2)
  - Generates unified parquet data and vocabulary
  - Saves results to shared storage

- **GPU Cluster**: Limited CPU cores (2), GPU access required
  - Runs TensorFlow operations (Steps 3-6)
  - Reads preprocessed data from shared storage
  - Performs training and generates submissions

## Shared Storage Requirements

The pipeline requires shared storage accessible from both clusters:
- `/shared/projects/belka_del/staging/` - CPU → GPU handoff area
- `/shared/projects/belka_del/results/` - GPU training outputs
- `/shared/projects/belka_del/logs/` - Centralized logging

## Step-by-Step Usage

### 1. CPU Cluster - Data Preprocessing

Run on the CPU cluster to preprocess data:

```bash
# Submit CPU preprocessing job
sbatch slurm/job_cpu_preprocess.sh

# Monitor job status
squeue -u $USER

# Check logs
tail -f logs/cpu_preprocess_<JOB_ID>.out
```

**Outputs saved to shared storage:**
- `belka.parquet` - Unified dataset
- `vocab.txt` - SMILES vocabulary
- `*.sha256` - Integrity checksums
- `cpu_processing_complete.marker` - Completion marker

### 2. Validate CPU Output (Optional)

Verify CPU preprocessing completed successfully:

```bash
./scripts/validate_data.sh cpu_output
```

### 3. GPU Cluster - Training

Run on the GPU cluster after CPU preprocessing completes:

```bash
# Submit GPU training job (choose training mode)
sbatch slurm/job_gpu_training.sh clf  # Classification
# OR
sbatch slurm/job_gpu_training.sh fps  # Fingerprint prediction  
# OR
sbatch slurm/job_gpu_training.sh mlm  # Masked language model

# Monitor job status
squeue -u $USER

# Check logs
tail -f logs/gpu_training_<JOB_ID>.out
```

**Outputs saved to shared storage:**
- `submission_<JOB_ID>.csv` - Final submission file
- `belka_*.model.keras` - Trained model
- `gpu_training_complete.marker` - Completion marker

### 4. Validate Full Pipeline (Optional)

Verify entire pipeline completed successfully:

```bash
./scripts/validate_data.sh all
```

## Configuration Files

- `configs/config_cpu.yaml` - CPU cluster configuration
- `configs/config_gpu.yaml` - GPU cluster configuration

Key differences:
- CPU config: High `n_workers` (16), disabled TensorFlow operations
- GPU config: Low `n_workers` (2), GPU optimizations enabled

## Data Flow

```
CPU Cluster                          GPU Cluster
-----------                          -----------
Raw Data                             
    ↓                                
Step 1: generate_parquet             
Step 2: get_vocab                    
    ↓                                
Save to Shared Storage  ──────────→  Read from Shared Storage
                                         ↓
                                     Step 3: make_dataset
                                     Step 4: create_val_dataset  
                                     Step 5: train_model
                                     Step 6: make_submission
                                         ↓
                                     Save Results to Shared Storage
```

## Error Troubleshooting

### CPU Cluster Issues

1. **Memory errors**: Reduce `chunk_size` in config_cpu.yaml
2. **File not found**: Ensure raw data files exist:
   - `data/raw/train.parquet`
   - `data/raw/test.parquet`  
   - `data/raw/DNA_Labeled_Data.csv`

### GPU Cluster Issues

1. **CPU preprocessing not found**: Run CPU cluster job first
2. **GPU memory errors**: Reduce `batch_size` in config_gpu.yaml
3. **TensorFlow GPU not detected**: Check CUDA/cuDNN modules

### Data Validation

Use the validation script to diagnose issues:

```bash
# Check CPU preprocessing
./scripts/validate_data.sh cpu_output

# Check GPU training  
./scripts/validate_data.sh gpu_output

# Check entire pipeline
./scripts/validate_data.sh all
```

## File Locations

**Input Data:**
- `data/raw/train.parquet`
- `data/raw/test.parquet`
- `data/raw/DNA_Labeled_Data.csv`

**Shared Storage:**
- `/shared/projects/belka_del/staging/` - CPU outputs
- `/shared/projects/belka_del/results/` - GPU outputs
- `/shared/projects/belka_del/logs/` - All logs

**Configuration:**
- `configs/config_cpu.yaml` - CPU cluster settings
- `configs/config_gpu.yaml` - GPU cluster settings

## Important Notes

1. **No Live Transfers**: The pipeline uses shared storage only - no live data transfers between clusters
2. **Data Integrity**: All files include SHA256 checksums for integrity validation
3. **Completion Markers**: Each stage creates completion markers to track progress
4. **Sequential Execution**: GPU cluster waits for CPU cluster completion marker
5. **Resource Optimization**: Each cluster uses configurations optimized for its hardware constraints

## Manual Pipeline Control

For advanced users, you can run individual steps manually:

```bash
# CPU cluster manual steps
poetry run python scripts/pipeline.py --cluster-type cpu --step preprocess

# GPU cluster manual steps  
poetry run python scripts/pipeline.py --cluster-type gpu --step tensorflow_pipeline --mode clf
poetry run python scripts/pipeline.py --cluster-type gpu --step make_submission --model-path path/to/model.keras
```