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
module load python/3.9    # Adjust based on your system
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

# Shared storage paths (must match CPU cluster configuration)
SHARED_STORAGE="/shared/projects/belka_del"
STAGING_DIR="$SHARED_STORAGE/staging"

echo "Current directory: $(pwd)"
echo "Available CPU cores: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
echo "Shared storage: $SHARED_STORAGE"
echo "Staging directory: $STAGING_DIR"

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
if [ ! -f "$STAGING_DIR/cpu_processing_complete.marker" ]; then
    echo "ERROR: CPU processing completion marker not found"
    echo "Expected: $STAGING_DIR/cpu_processing_complete.marker"
    echo ""
    echo "CPU preprocessing must complete first. Run:"
    echo "sbatch slurm/job_cpu_preprocess.sh"
    echo ""
    echo "Or check if staging directory path is correct:"
    echo "STAGING_DIR=$STAGING_DIR"
    exit 1
fi

echo "✓ CPU processing completion marker found"
echo "Completion marker contents:"
cat "$STAGING_DIR/cpu_processing_complete.marker"
echo ""

# Verify required input files from CPU preprocessing
echo "Checking for required input files from CPU preprocessing..."
REQUIRED_FILES=("belka.parquet" "vocab.txt" "belka.parquet.sha256" "vocab.txt.sha256")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$STAGING_DIR/$file" ]; then
        echo "ERROR: Required file not found: $STAGING_DIR/$file"
        echo "CPU preprocessing may have failed or files were not properly staged"
        exit 1
    fi
done

echo "✓ All required files found in staging area"

# Verify data integrity using checksums
echo "Verifying data integrity..."
cd $STAGING_DIR

if sha256sum -c belka.parquet.sha256; then
    echo "✓ belka.parquet integrity verified"
else
    echo "✗ belka.parquet integrity check failed"
    exit 1
fi

if sha256sum -c vocab.txt.sha256; then
    echo "✓ vocab.txt integrity verified"
else
    echo "✗ vocab.txt integrity check failed"
    exit 1
fi

cd $PROJECT_DIR

# Copy files from staging to local working directory
echo "Copying files from staging to local working directory..."
cp "$STAGING_DIR/belka.parquet" data/raw/
cp "$STAGING_DIR/vocab.txt" data/raw/

# Verify local copies
if [ ! -f "data/raw/belka.parquet" ] || [ ! -f "data/raw/vocab.txt" ]; then
    echo "ERROR: Failed to copy files from staging to local directory"
    exit 1
fi

echo "✓ Files successfully copied to local working directory"
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
        
        # Save results to shared storage
        echo "Saving results to shared storage..."
        mkdir -p $SHARED_STORAGE/results
        cp "$LATEST_MODEL" $SHARED_STORAGE/results/
        if [ -f "data/raw/submission.csv" ]; then
            cp data/raw/submission.csv $SHARED_STORAGE/results/submission_${SLURM_JOB_ID}.csv
        fi
        cp -r models checkpoints $SHARED_STORAGE/results/ 2>/dev/null || true
        
        # Create completion marker
        cat > $SHARED_STORAGE/gpu_training_complete.marker << EOF
# Belka GPU Training Completion Marker
job_id: $SLURM_JOB_ID
completion_time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
node: $SLURMD_NODENAME
training_mode: $TRAINING_MODE
model_path: $LATEST_MODEL
model_size_bytes: $(stat -c%s "$LATEST_MODEL")
submission_generated: $([ -f "data/raw/submission.csv" ] && echo "true" || echo "false")
results_saved: $SHARED_STORAGE/results/
EOF
        
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

# Copy logs to shared storage
cp logs/gpu_training_${SLURM_JOB_ID}.log $SHARED_STORAGE/logs/

# Final summary
echo "=============================================="
echo "GPU TRAINING COMPLETED SUCCESSFULLY"
echo "Training mode: $TRAINING_MODE"
echo "Final model: $LATEST_MODEL"
echo "Results saved to: $SHARED_STORAGE/results/"
echo "End time: $(date)"
echo "Total job duration: $(squeue -j $SLURM_JOB_ID -h -o %M 2>/dev/null || echo 'N/A')"
echo "=============================================="