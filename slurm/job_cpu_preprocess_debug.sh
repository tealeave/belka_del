#!/bin/bash
#SBATCH --job-name=belka_debug
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/cpu_preprocess_debug_%j.out
#SBATCH --error=logs/cpu_preprocess_debug_%j.err

# Belka Transformer - DEBUG Mode CPU Preprocessing
# Quick testing with minimal resources - processes only 3 chunks (~30 seconds)
# Perfect for development, testing fixes, and code validation

# Initialize Python utilities and job tracking
poetry run python scripts/slurm_utils.py --cluster-type cpu job-start "belka_preprocess_debug"

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

# Memory management (reduced for debug mode)
export MALLOC_ARENA_MAX=2

# Project directory
PROJECT_DIR="/pub/ddlin/projects/belka_del"
cd $PROJECT_DIR

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
    poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess_debug" --exit-code 1 --error-message "Cleanup failed - existing output files detected"
    exit 1
fi

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
    
    # Verify output files using Python utilities
    echo "Verifying output files..."
    poetry run python scripts/slurm_utils.py --cluster-type cpu validate-outputs
    if [ $? -ne 0 ]; then
        echo "✗ Output file validation failed"
        poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess_debug" --exit-code 1 --error-message "Output validation failed"
        exit 1
    fi
    
    echo "=============================================="
    echo "🧪 DEBUG PREPROCESSING COMPLETED SUCCESSFULLY"
    echo "Files created (debug subset):"
    echo "- belka.parquet: $(du -sh data/raw/belka.parquet | cut -f1)"
    echo "- vocab.txt: $(du -h data/raw/vocab.txt | cut -f1) ($(wc -l < data/raw/vocab.txt) tokens)"
    echo ""
    echo "⚡ Debug mode processed ~96K rows in ~30 seconds"
    echo "📝 For full production run: sbatch slurm/job_cpu_preprocess.sh"
    echo "=============================================="
    
    # Log successful job completion
    poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess_debug" --exit-code 0 --save-report
    
else
    echo "✗ DEBUG preprocessing failed with exit code $?"
    echo "Check logs/cpu_preprocess_debug_${SLURM_JOB_ID}.log for details"
    
    # Log failed job completion
    poetry run python scripts/slurm_utils.py --cluster-type cpu job-end "belka_preprocess_debug" --exit-code $? --error-message "DEBUG preprocessing pipeline failed" --save-report
    exit 1
fi