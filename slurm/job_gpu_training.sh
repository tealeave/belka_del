#!/bin/bash
#SBATCH --job-name=belka_gpu_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpu_training_%j.out
#SBATCH --error=logs/gpu_training_%j.err

# Belka Transformer - GPU Cluster Training Job
# Runs Steps 3-6: TFRecord Dataset + Training + Submission
# Reads preprocessed data from shared storage staging area

echo "=============================================="
echo "Belka Transformer - GPU Cluster Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

# Environment setup
module load python/3.10.2
module load cuda/11.8     # Adjust CUDA version as needed
module load cudnn/8.6     # Adjust cuDNN version as needed

# Create logs directory
mkdir -p logs

# Set environment variables for GPU optimization
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# CPU limitations handling (only 2 cores available)
export OMP_NUM_THREADS=2
export NUMEXPR_MAX_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2

# Memory management
export TF_GPU_MEMORY_ALLOCATION=cuda_malloc_async

# Project directory
PROJECT_DIR="/pub/ddlin/projects/belka_del"
cd $PROJECT_DIR

# All processing uses local repository files

echo "Current directory: $(pwd)"
echo "Available CPU cores: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
echo "Processing locally from data/raw/ directory"

# GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Check TensorFlow GPU setup
echo "Checking TensorFlow GPU setup..."
poetry run python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
if tf.config.list_physical_devices('GPU'):
    print('✓ GPU detected and available')
else:
    print('✗ No GPU detected')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: TensorFlow GPU setup failed"
    exit 1
fi

# Check for CPU preprocessing completion
echo "Checking for CPU preprocessing completion..."
if [ ! -f "data/raw/belka.parquet" ] || [ ! -f "data/raw/vocab.txt" ]; then
    echo "ERROR: Required preprocessing files not found"
    echo "Expected files:"
    echo "  - data/raw/belka.parquet"
    echo "  - data/raw/vocab.txt"
    echo ""
    echo "CPU preprocessing must complete first. Run:"
    echo "sbatch slurm/job_cpu_preprocess.sh"
    exit 1
fi

echo "✓ CPU preprocessing files found locally"

# Verify required input files are available locally
echo "Verifying required input files..."
if [ ! -f "data/raw/belka.parquet" ] || [ ! -f "data/raw/vocab.txt" ]; then
    echo "ERROR: Required files not found in data/raw/"
    echo "Expected files:"
    echo "  - data/raw/belka.parquet"
    echo "  - data/raw/vocab.txt"
    exit 1
fi

echo "✓ All required files found locally"
echo "Local file sizes:"
echo "- belka.parquet: $(du -h data/raw/belka.parquet | cut -f1)"
echo "- vocab.txt: $(du -h data/raw/vocab.txt | cut -f1) ($(wc -l < data/raw/vocab.txt) tokens)"

# Set training mode (default to classification)
TRAINING_MODE=${1:-clf}
echo "Training mode: $TRAINING_MODE"

# Create output directories
mkdir -p models checkpoints results

# Run TensorFlow pipeline
echo "=============================================="
echo "Starting TensorFlow pipeline on GPU cluster..."
echo "Steps: make_dataset → create_val_dataset → train_model"
echo "=============================================="

time poetry run python scripts/pipeline.py \
    --cluster-type gpu \
    --step tensorflow_pipeline \
    --mode $TRAINING_MODE \
    --log-level INFO \
    --log-file logs/gpu_training_${SLURM_JOB_ID}.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✓ TensorFlow pipeline completed successfully!"
    
    # Verify output files
    echo "Verifying output files..."
    
    if [ -f "data/raw/belka.tfr" ]; then
        echo "✓ belka.tfr created"
    else
        echo "✗ belka.tfr not found"
    fi
    
    if [ -f "data/raw/belka_val.tfr" ]; then
        echo "✓ belka_val.tfr created"
    else
        echo "✗ belka_val.tfr not found"
    fi
    
    # Find the latest trained model
    LATEST_MODEL=$(find . -name "belka_*.model.keras" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_MODEL" ]; then
        echo "✓ Trained model found: $LATEST_MODEL"
        echo "Model size: $(du -h "$LATEST_MODEL" | cut -f1)"
        
        # Generate submission
        echo "Generating submission..."
        poetry run python scripts/pipeline.py \
            --cluster-type gpu \
            --step make_submission \
            --model-path "$LATEST_MODEL" \
            --log-level INFO
        
        if [ $? -eq 0 ] && [ -f "data/raw/submission.csv" ]; then
            echo "✓ Submission generated: data/raw/submission.csv"
            echo "Submission file size: $(du -h data/raw/submission.csv | cut -f1)"
            echo "Submission entries: $(wc -l < data/raw/submission.csv) lines"
            echo ""
            echo "Submission preview:"
            head -5 data/raw/submission.csv
        else
            echo "✗ Submission generation failed"
        fi
        
        # Results are already saved locally in the repository
        echo "Training results saved locally:"
        echo "- Model: $LATEST_MODEL"
        echo "- Models directory: models/"
        echo "- Checkpoints directory: checkpoints/"
        if [ -f "data/raw/submission.csv" ]; then
            echo "- Submission: data/raw/submission.csv"
        fi
        
    else
        echo "✗ No trained model found"
        exit 1
    fi
    
else
    echo "✗ TensorFlow pipeline failed with exit code $?"
    
    # Debugging information
    echo "=============================================="
    echo "DEBUGGING INFORMATION"
    echo "=============================================="
    echo "GPU memory usage:"
    nvidia-smi
    
    echo "Disk usage:"
    df -h .
    
    echo "Recent log entries:"
    if [ -f "logs/gpu_training_${SLURM_JOB_ID}.log" ]; then
        tail -20 logs/gpu_training_${SLURM_JOB_ID}.log
    fi
    
    exit 1
fi

# Logs are kept locally in logs/ directory

# Final summary
echo "=============================================="
echo "GPU TRAINING COMPLETED SUCCESSFULLY"
echo "Training mode: $TRAINING_MODE"
echo "Final model: $LATEST_MODEL"
echo "Results saved locally in repository"
echo "End time: $(date)"
echo "Total job duration: $(squeue -j $SLURM_JOB_ID -h -o %M 2>/dev/null || echo 'N/A')"
echo "=============================================="