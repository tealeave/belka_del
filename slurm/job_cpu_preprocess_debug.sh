#!/bin/bash
#SBATCH --job-name=belka_debug
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/cpu_preprocess_debug_%j.out
#SBATCH --error=logs/cpu_preprocess_debug_%j.err

# Belka Transformer - DEBUG Mode CPU Preprocessing
# Quick testing with minimal resources - processes only 3 chunks (~30 seconds)
# Perfect for development, testing fixes, and code validation

echo "=============================================="
echo "Belka Transformer - DEBUG MODE Preprocessing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "⚠️  DEBUG MODE: Processing only 3 chunks for quick testing"
echo "=============================================="

# Environment setup
module load python/3.10.2
# GCC not required for Poetry environment

# Activate Poetry virtual environment
echo "Activating Poetry virtual environment..."
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv directory not found. Make sure Poetry environment is set up."
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Verify Python version
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Poetry version: $(poetry --version)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for optimal CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable TensorFlow GPU detection on CPU cluster
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=1

# Memory management (reduced for debug mode)
export MALLOC_ARENA_MAX=2

# Project directory
PROJECT_DIR="/pub/ddlin/projects/belka_del"
cd $PROJECT_DIR

echo "Current directory: $(pwd)"
echo "Available CPU cores: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
echo "Processing locally within repository"

# Ensure local data directories exist
mkdir -p data/raw
mkdir -p data/processed

# Check for required data files
echo "Checking for required input files..."
if [ ! -f "data/raw/train.parquet" ]; then
    echo "ERROR: data/raw/train.parquet not found"
    exit 1
fi
if [ ! -f "data/raw/test.parquet" ]; then
    echo "ERROR: data/raw/test.parquet not found"
    exit 1
fi
if [ ! -f "data/raw/DNA_Labeled_Data.csv" ]; then
    echo "ERROR: data/raw/DNA_Labeled_Data.csv not found"
    exit 1
fi
echo "All required input files found."

# Clean any previous processed data
echo "Cleaning previous processed data..."
rm -f data/raw/belka.parquet
rm -f data/raw/vocab.txt

# Run DEBUG preprocessing pipeline
echo "Starting DEBUG preprocessing pipeline..."
echo "⚡ DEBUG Mode: Fast testing (3 chunks only)"

time poetry run python scripts/pipeline.py \
    --step preprocess \
    --debug \
    --log-level INFO \
    --log-file logs/cpu_preprocess_debug_${SLURM_JOB_ID}.log

# Check if preprocessing completed successfully
if [ $? -eq 0 ]; then
    echo "✅ DEBUG preprocessing completed successfully!"
    
    # Verify output files exist locally
    echo "Verifying local output files..."
    if [ -f "data/raw/belka.parquet" ]; then
        echo "✓ belka.parquet created ($(du -h data/raw/belka.parquet | cut -f1))"
    else
        echo "✗ belka.parquet not found"
        exit 1
    fi
    
    if [ -f "data/raw/vocab.txt" ]; then
        echo "✓ vocab.txt created ($(wc -l < data/raw/vocab.txt) tokens)"
    else
        echo "✗ vocab.txt not found"
        exit 1
    fi
    
    echo "=============================================="
    echo "🧪 DEBUG PREPROCESSING COMPLETED SUCCESSFULLY"
    echo "Files created (debug subset):"
    echo "- belka.parquet: $(du -h data/raw/belka.parquet | cut -f1)"
    echo "- vocab.txt: $(du -h data/raw/vocab.txt | cut -f1) ($(wc -l < data/raw/vocab.txt) tokens)"
    echo ""
    echo "⚡ Debug mode processed ~96K rows in ~30 seconds"
    echo "📝 For full production run: sbatch slurm/job_cpu_preprocess.sh"
    echo "=============================================="
    
else
    echo "✗ DEBUG preprocessing failed with exit code $?"
    echo "Check logs/cpu_preprocess_debug_${SLURM_JOB_ID}.log for details"
    exit 1
fi

echo "End time: $(date)"