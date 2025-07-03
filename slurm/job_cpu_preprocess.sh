#!/bin/bash
#SBATCH --job-name=belka_preprocess
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=logs/cpu_preprocess_%j.out
#SBATCH --error=logs/cpu_preprocess_%j.err

# Belka Transformer - CPU Cluster Preprocessing Job
# Runs Steps 1-2: Generate Parquet Data + Extract Vocabulary
# Saves results to shared storage for GPU cluster pickup
#
# Usage:
#   sbatch slurm/job_cpu_preprocess.sh                    # Full processing
#   DEBUG_MODE=true sbatch slurm/job_cpu_preprocess.sh    # Debug mode (3 chunks)
#   sbatch slurm/job_cpu_preprocess_debug.sh              # Dedicated debug script

# Initialize Python utilities and job tracking
poetry run python scripts/slurm_utils.py --cluster-type cpu job-start "belka_preprocess"

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

# Create standard directories using Python utilities
poetry run python scripts/slurm_utils.py --cluster-type cpu create-dirs

# Set environment variables for optimal CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable TensorFlow GPU detection on CPU cluster
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=1

# Memory management
export MALLOC_ARENA_MAX=4

# Project directory
PROJECT_DIR="/pub/ddlin/projects/belka_del"
cd $PROJECT_DIR

# All processing will be done locally within the repository

echo "Current directory: $(pwd)"
echo "Available CPU cores: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
echo "Processing locally within repository"

# Directories already created by Python utilities above

# Validate required input files using Python utilities
poetry run python scripts/slurm_utils.py --cluster-type cpu validate-inputs
if [ $? -ne 0 ]; then
    echo "ERROR: Input file validation failed"
    exit 1
fi

# Clean previous processed data safely using Python utilities
poetry run python scripts/slurm_utils.py --cluster-type cpu clean
if [ $? -ne 0 ]; then
    echo "ERROR: Cleanup failed - check logs for details"
    poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess" --exit-code 1 --error-message "Cleanup failed - existing output files detected"
    exit 1
fi

# Run preprocessing pipeline
echo "Starting preprocessing pipeline..."

# Check if DEBUG mode is requested via environment variable
DEBUG_FLAG=""
if [ "${DEBUG_MODE:-false}" = "true" ]; then
    DEBUG_FLAG="--debug"
    echo "🧪 DEBUG MODE ENABLED: Processing only 3 chunks for quick testing"
    echo "Step: preprocess (DEBUG mode - fast testing)"
else
    echo "Step: preprocess (CPU cluster optimized - full processing)"
fi

time poetry run python scripts/pipeline.py \
    --step preprocess \
    ${DEBUG_FLAG} \
    --log-level INFO \
    --log-file logs/cpu_preprocess_${SLURM_JOB_ID}.log

# Check if preprocessing completed successfully
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully!"
    
    # Verify output files using Python utilities
    echo "Verifying output files..."
    poetry run python scripts/slurm_utils.py --cluster-type cpu validate-outputs
    if [ $? -ne 0 ]; then
        echo "✗ Output file validation failed"
        poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess" --exit-code 1 --error-message "Output validation failed"
        exit 1
    fi
    
    # Verify files are ready for GPU processing
    echo "Verifying local files for GPU processing..."
    echo "Local file sizes:"
    echo "- belka.parquet: $(du -sh data/raw/belka.parquet | cut -f1)"
    echo "- vocab.txt: $(du -h data/raw/vocab.txt | cut -f1) ($(wc -l < data/raw/vocab.txt) tokens)"
    
    echo "=============================================="
    if [ "${DEBUG_MODE:-false}" = "true" ]; then
        echo "🧪 DEBUG PREPROCESSING COMPLETED SUCCESSFULLY"
        echo "Debug files created (3 chunks processed):"
        echo "- belka.parquet: $(du -sh data/raw/belka.parquet | cut -f1)"
        echo "- vocab.txt: $(du -h data/raw/vocab.txt | cut -f1) ($(wc -l < data/raw/vocab.txt) tokens)"
        echo ""
        echo "⚡ Debug mode processed ~96K rows for quick validation"
        echo "📝 For full production run: sbatch slurm/job_cpu_preprocess.sh"
    else
        echo "CPU preprocessing COMPLETED SUCCESSFULLY"
        echo "Files ready for GPU processing:"
        echo "- data/raw/belka.parquet ($(du -sh data/raw/belka.parquet | cut -f1))"
        echo "- data/raw/vocab.txt ($(du -h data/raw/vocab.txt | cut -f1))"
        echo ""
        echo "NEXT STEP: Submit GPU training job"
        echo "Command: sbatch slurm/job_gpu_training.sh"
    fi
    echo "=============================================="
    
    # Log successful job completion
    poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess" --exit-code 0 --save-report
    
else
    echo "✗ Preprocessing failed with exit code $?"
    echo "Check logs/cpu_preprocess_${SLURM_JOB_ID}.log for details"
    
    # Log failed job completion
    poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess" --exit-code $? --error-message "Preprocessing pipeline failed" --save-report
    exit 1
fi