#!/bin/bash
#SBATCH --job-name=belka_inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output=logs/gpu_inference_%j.out
#SBATCH --error=logs/gpu_inference_%j.err

# Belka Transformer - GPU Cluster Inference Job
# Runs submission generation only (assumes model is already trained)

echo "=============================================="
echo "Belka Transformer - GPU Cluster Inference"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Environment setup
module load python/3.10.2
module load cuda/11.8
module load cudnn/8.6

# Create logs directory
mkdir -p logs

# GPU environment
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TF_CPP_MIN_LOG_LEVEL=1

# Model path (required parameter)
MODEL_PATH=${1:-""}
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model path not provided"
    echo "Usage: sbatch job_gpu_inference.sh <model_path>"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Using model: $MODEL_PATH"

# Project directory
PROJECT_DIR="/pub/ddlin/projects/belka_del"
cd $PROJECT_DIR

# Generate submission
echo "Generating submission..."
time poetry run python scripts/pipeline.py \
    --cluster-type gpu \
    --step make_submission \
    --model-path "$MODEL_PATH" \
    --log-level INFO \
    --log-file logs/gpu_inference_${SLURM_JOB_ID}.log

if [ $? -eq 0 ] && [ -f "data/raw/submission.csv" ]; then
    echo "✓ Submission generated successfully"
    echo "File: data/raw/submission.csv"
    echo "Size: $(du -h data/raw/submission.csv | cut -f1)"
    echo "Preview:"
    head -5 data/raw/submission.csv
else
    echo "✗ Submission generation failed"
    exit 1
fi

echo "=============================================="
echo "Inference job completed"
echo "End time: $(date)"
echo "=============================================="