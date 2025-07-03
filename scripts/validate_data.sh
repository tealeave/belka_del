#!/bin/bash

# Belka Transformer - Data Validation Utility
# Validates data integrity and readiness in local repository
# Validates locally stored data files

echo "=============================================="
echo "Belka Transformer - Data Validation Utility"
echo "=============================================="

# Configuration - Local repository paths
PROJECT_DIR="/pub/ddlin/projects/belka_del"
DATA_DIR="$PROJECT_DIR/data/raw"
MODELS_DIR="$PROJECT_DIR/models"
RESULTS_DIR="$PROJECT_DIR/results"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints"

# Validation mode
VALIDATION_MODE=${1:-cpu_output}

# Validate mode
if [[ "$VALIDATION_MODE" != "cpu_output" && "$VALIDATION_MODE" != "gpu_output" && "$VALIDATION_MODE" != "all" ]]; then
    echo "ERROR: Invalid validation mode. Use 'cpu_output', 'gpu_output', or 'all'"
    echo "Usage: $0 [cpu_output|gpu_output|all]"
    exit 1
fi

echo "Validation mode: $VALIDATION_MODE"
echo "Project directory: $PROJECT_DIR"

case $VALIDATION_MODE in
    "cpu_output")
        echo "Validating CPU preprocessing outputs..."
        
        # Check if required files exist locally
        echo "Checking CPU preprocessing outputs..."
        if [ ! -d "$DATA_DIR/belka.parquet" ] || [ ! -f "$DATA_DIR/vocab.txt" ]; then
            echo "ERROR: Required preprocessing files not found"
            echo "Expected files:"
            echo "  - $DATA_DIR/belka.parquet (directory)"
            echo "  - $DATA_DIR/vocab.txt"
            echo ""
            echo "CPU preprocessing must complete first. Run:"
            echo "sbatch slurm/job_cpu_preprocess.sh"
            exit 1
        fi
        
        echo "✓ CPU preprocessing output files found locally"
        
        # Required files from CPU preprocessing
        REQUIRED_FILES=("belka.parquet" "vocab.txt")
        
        # Check if files exist and get information
        echo "Verifying required files exist..."
        for file in "${REQUIRED_FILES[@]}"; do
            if [ "$file" = "belka.parquet" ]; then
                if [ ! -d "$DATA_DIR/$file" ]; then
                    echo "ERROR: Required directory not found: $DATA_DIR/$file"
                    echo "CPU preprocessing may have failed"
                    exit 1
                fi
            else
                if [ ! -f "$DATA_DIR/$file" ]; then
                    echo "ERROR: Required file not found: $DATA_DIR/$file"
                    echo "CPU preprocessing may have failed"
                    exit 1
                fi
            fi
        done
        echo "✓ All required files found locally"
        
        # Verify basic file integrity (non-empty, readable)
        echo "Verifying file integrity..."
        if [ -d "$DATA_DIR/belka.parquet" ] && [ -r "$DATA_DIR/belka.parquet" ]; then
            echo "✓ belka.parquet directory is accessible"
        else
            echo "✗ belka.parquet directory integrity check failed"
            exit 1
        fi
        
        if [ -s "$DATA_DIR/vocab.txt" ] && [ -r "$DATA_DIR/vocab.txt" ]; then
            echo "✓ vocab.txt is accessible and non-empty"
        else
            echo "✗ vocab.txt integrity check failed"
            exit 1
        fi
        
        # Display file information
        echo ""
        echo "File information:"
        echo "- belka.parquet: $(du -sh $DATA_DIR/belka.parquet | cut -f1) (directory)"
        echo "- vocab.txt: $(du -h $DATA_DIR/vocab.txt | cut -f1) ($(wc -l < $DATA_DIR/vocab.txt) tokens)"
        
        echo ""
        echo "✓ CPU output validation successful"
        echo "GPU cluster is ready to process these files"
        echo "Next step: sbatch slurm/job_gpu_training.sh [clf|fps|mlm]"
        ;;
        
    "gpu_output")
        echo "Validating GPU training outputs..."
        
        # Check if GPU training completed by looking for model files
        echo "Checking GPU training completion..."
        MODEL_COUNT=$(find "$PROJECT_DIR" -name "*.keras" 2>/dev/null | wc -l)
        if [ $MODEL_COUNT -eq 0 ]; then
            echo "ERROR: No trained model files found"
            echo "Expected: *.keras files in project directory"
            echo ""
            echo "GPU training must complete first. Run:"
            echo "sbatch slurm/job_gpu_training.sh [clf|fps|mlm]"
            exit 1
        fi
        
        echo "✓ Found $MODEL_COUNT trained model file(s)"
        
        # Check for output directories and files
        echo "Checking output directories..."
        if [ -d "$MODELS_DIR" ]; then
            echo "✓ Models directory found: $MODELS_DIR"
        fi
        if [ -d "$CHECKPOINTS_DIR" ]; then
            echo "✓ Checkpoints directory found: $CHECKPOINTS_DIR"
        fi
        if [ -d "$RESULTS_DIR" ]; then
            echo "✓ Results directory found: $RESULTS_DIR"
            echo "Results directory contents:"
            ls -la "$RESULTS_DIR/" 2>/dev/null || echo "(empty)"
        fi
        
        # Check for submission file
        if [ -f "$DATA_DIR/submission.csv" ]; then
            echo "✓ Submission file found: $DATA_DIR/submission.csv"
            echo "Submission file size: $(du -h "$DATA_DIR/submission.csv" | cut -f1)"
            echo "Submission entries: $(wc -l < "$DATA_DIR/submission.csv") lines"
            echo ""
            echo "Submission preview:"
            head -5 "$DATA_DIR/submission.csv"
        else
            echo "⚠ No submission file found in data/raw/"
        fi
        
        # Check for model files throughout project
        MODEL_FILES=$(find "$PROJECT_DIR" -name "*.keras" | wc -l)
        if [ $MODEL_FILES -gt 0 ]; then
            echo "✓ Found $MODEL_FILES model file(s)"
            find "$PROJECT_DIR" -name "*.keras" -exec du -h {} \;
        else
            echo "⚠ No model files found in project"
        fi
        
        echo ""
        echo "✓ GPU output validation completed"
        ;;
        
    "all")
        echo "Validating all pipeline outputs..."
        echo ""
        
        # Validate CPU output first
        $0 cpu_output
        CPU_STATUS=$?
        
        echo ""
        echo "=============================================="
        echo ""
        
        # Validate GPU output
        $0 gpu_output  
        GPU_STATUS=$?
        
        echo ""
        echo "=============================================="
        echo "FULL PIPELINE VALIDATION SUMMARY"
        echo "=============================================="
        
        if [ $CPU_STATUS -eq 0 ]; then
            echo "✓ CPU preprocessing: PASSED"
        else
            echo "✗ CPU preprocessing: FAILED"
        fi
        
        if [ $GPU_STATUS -eq 0 ]; then
            echo "✓ GPU training: PASSED"
        else
            echo "✗ GPU training: FAILED"
        fi
        
        if [ $CPU_STATUS -eq 0 ] && [ $GPU_STATUS -eq 0 ]; then
            echo ""
            echo "🎉 Full pipeline validation successful!"
            echo "All data properly saved and validated in local repository"
        else
            echo ""
            echo "❌ Pipeline validation failed"
            echo "Check individual component logs for details"
            exit 1
        fi
        ;;
esac

echo "=============================================="
echo "Data validation completed successfully"
echo "Mode: $VALIDATION_MODE"
echo "=============================================="