#!/bin/bash
#SBATCH --job-name=belka_preprocess
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --output=logs/cpu_preprocess_%j.out
#SBATCH --error=logs/cpu_preprocess_%j.err

# Belka Transformer - CPU Cluster Preprocessing Job
# Runs Steps 1-2: Generate Parquet Data + Extract Vocabulary
# Saves results to shared storage for GPU cluster pickup

echo "=============================================="
echo "Belka Transformer - CPU Cluster Preprocessing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Environment setup
module load python/3.9  # Adjust based on your system
module load gcc/9.3.0   # Required for some dependencies

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

# Memory management
export MALLOC_ARENA_MAX=4

# Project directory
PROJECT_DIR="/pub/ddlin/projects/belka_del"
cd $PROJECT_DIR

# Shared storage paths (modify these for your HPC setup)
SHARED_STORAGE="/shared/projects/belka_del"
STAGING_DIR="$SHARED_STORAGE/staging"

echo "Current directory: $(pwd)"
echo "Available CPU cores: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE}MB"
echo "Shared storage: $SHARED_STORAGE"
echo "Staging directory: $STAGING_DIR"

# Create shared storage directories
mkdir -p $SHARED_STORAGE
mkdir -p $STAGING_DIR
mkdir -p $SHARED_STORAGE/logs

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

# Clean any previous staging data
echo "Cleaning staging directory..."
rm -f $STAGING_DIR/belka.parquet*
rm -f $STAGING_DIR/vocab.txt*
rm -f $STAGING_DIR/cpu_processing_complete.marker

# Run preprocessing pipeline
echo "Starting preprocessing pipeline..."
echo "Step: preprocess (CPU cluster optimized)"

time poetry run python scripts/pipeline.py \
    --cluster-type cpu \
    --step preprocess \
    --log-level INFO \
    --log-file logs/cpu_preprocess_${SLURM_JOB_ID}.log

# Check if preprocessing completed successfully
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully!"
    
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
    
    # Copy files to shared storage/staging area
    echo "Copying files to staging area for GPU cluster..."
    cp data/raw/belka.parquet $STAGING_DIR/
    cp data/raw/vocab.txt $STAGING_DIR/
    
    # Calculate checksums for data integrity validation
    echo "Calculating checksums for data integrity..."
    cd $STAGING_DIR
    sha256sum belka.parquet > belka.parquet.sha256
    sha256sum vocab.txt > vocab.txt.sha256
    cd $PROJECT_DIR
    
    # Create completion marker with metadata
    echo "Creating completion marker..."
    cat > $STAGING_DIR/cpu_processing_complete.marker << EOF
# Belka CPU Processing Completion Marker
job_id: $SLURM_JOB_ID
completion_time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
node: $SLURMD_NODENAME
files_ready:
  - belka.parquet
  - vocab.txt
  - belka.parquet.sha256
  - vocab.txt.sha256
file_sizes:
  belka_parquet_bytes: $(stat -c%s $STAGING_DIR/belka.parquet)
  vocab_txt_bytes: $(stat -c%s $STAGING_DIR/vocab.txt)
checksums:
  belka_parquet_sha256: $(sha256sum $STAGING_DIR/belka.parquet | cut -d' ' -f1)
  vocab_txt_sha256: $(sha256sum $STAGING_DIR/vocab.txt | cut -d' ' -f1)
next_step: "GPU cluster can now run tensorflow_pipeline"
EOF
    
    # Verify staging area contents
    echo "Staging area contents:"
    ls -la $STAGING_DIR/
    
    # Copy logs to shared storage
    cp logs/cpu_preprocess_${SLURM_JOB_ID}.log $SHARED_STORAGE/logs/
    
    echo "=============================================="
    echo "CPU preprocessing COMPLETED SUCCESSFULLY"
    echo "Files staged for GPU cluster:"
    echo "- $STAGING_DIR/belka.parquet ($(du -h $STAGING_DIR/belka.parquet | cut -f1))"
    echo "- $STAGING_DIR/vocab.txt ($(du -h $STAGING_DIR/vocab.txt | cut -f1))"
    echo "- Checksums and completion marker created"
    echo ""
    echo "NEXT STEP: Submit GPU cluster job"
    echo "Command: sbatch slurm/job_gpu_training.sh"
    echo "=============================================="
    
else
    echo "✗ Preprocessing failed with exit code $?"
    echo "Check logs/cpu_preprocess_${SLURM_JOB_ID}.log for details"
    exit 1
fi

echo "End time: $(date)"