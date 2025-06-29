#!/bin/bash

# Belka Transformer - Data Validation Utility
# Validates data integrity and readiness in shared storage
# NO LIVE TRANSFERS - Only validates saved data

echo "=============================================="
echo "Belka Transformer - Data Validation Utility"
echo "=============================================="

# Configuration - MODIFY THESE FOR YOUR HPC SETUP
SHARED_STORAGE="/shared/projects/belka_del"
STAGING_DIR="$SHARED_STORAGE/staging"
RESULTS_DIR="$SHARED_STORAGE/results"

# Validation mode
VALIDATION_MODE=${1:-cpu_output}

# Validate mode
if [[ "$VALIDATION_MODE" != "cpu_output" && "$VALIDATION_MODE" != "gpu_output" && "$VALIDATION_MODE" != "all" ]]; then
    echo "ERROR: Invalid validation mode. Use 'cpu_output', 'gpu_output', or 'all'"
    echo "Usage: $0 [cpu_output|gpu_output|all]"
    exit 1
fi

echo "Validation mode: $VALIDATION_MODE"
echo "Shared storage: $SHARED_STORAGE"

case $VALIDATION_MODE in
    "cpu_output")
        echo "Validating CPU preprocessing outputs..."
        
        # Check if completion marker exists
        echo "Checking CPU preprocessing completion..."
        if [ ! -f "$STAGING_DIR/cpu_processing_complete.marker" ]; then
            echo "ERROR: CPU processing completion marker not found"
            echo "Expected: $STAGING_DIR/cpu_processing_complete.marker"
            echo ""
            echo "CPU preprocessing must complete first. Run:"
            echo "sbatch slurm/job_cpu_preprocess.sh"
            exit 1
        fi
        
        echo "✓ CPU preprocessing completion marker found"
        echo "Completion marker contents:"
        cat "$STAGING_DIR/cpu_processing_complete.marker"
        echo ""
        
        # Required files from CPU preprocessing
        REQUIRED_FILES=("belka.parquet" "vocab.txt" "belka.parquet.sha256" "vocab.txt.sha256")
        
        # Check if files exist
        echo "Verifying required files exist..."
        for file in "${REQUIRED_FILES[@]}"; do
            if [ ! -f "$STAGING_DIR/$file" ]; then
                echo "ERROR: Required file not found: $STAGING_DIR/$file"
                echo "CPU preprocessing may have failed or files were not properly saved"
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
        
        # Display file information
        echo ""
        echo "File information:"
        echo "- belka.parquet: $(du -h belka.parquet | cut -f1)"
        echo "- vocab.txt: $(du -h vocab.txt | cut -f1) ($(wc -l < vocab.txt) tokens)"
        
        echo ""
        echo "✓ CPU output validation successful"
        echo "GPU cluster is ready to process these files"
        echo "Next step: sbatch slurm/job_gpu_training.sh [clf|fps|mlm]"
        ;;
        
    "gpu_output")
        echo "Validating GPU training outputs..."
        
        # Check if GPU training completed
        echo "Checking GPU training completion..."
        if [ ! -f "$SHARED_STORAGE/gpu_training_complete.marker" ]; then
            echo "ERROR: GPU training completion marker not found"
            echo "Expected: $SHARED_STORAGE/gpu_training_complete.marker"
            echo ""
            echo "GPU training must complete first. Run:"
            echo "sbatch slurm/job_gpu_training.sh [clf|fps|mlm]"
            exit 1
        fi
        
        echo "✓ GPU training completion marker found"
        echo "Completion marker contents:"
        cat "$SHARED_STORAGE/gpu_training_complete.marker"
        echo ""
        
        # Check results directory
        if [ ! -d "$RESULTS_DIR" ]; then
            echo "ERROR: Results directory not found: $RESULTS_DIR"
            exit 1
        fi
        
        echo "Results directory contents:"
        ls -la "$RESULTS_DIR/"
        
        # Check for submission file
        SUBMISSION_FILE=$(find "$RESULTS_DIR" -name "submission_*.csv" | head -1)
        if [ -n "$SUBMISSION_FILE" ]; then
            echo "✓ Submission file found: $SUBMISSION_FILE"
            echo "Submission file size: $(du -h "$SUBMISSION_FILE" | cut -f1)"
            echo "Submission entries: $(wc -l < "$SUBMISSION_FILE") lines"
            echo ""
            echo "Submission preview:"
            head -5 "$SUBMISSION_FILE"
        else
            echo "⚠ No submission file found in results"
        fi
        
        # Check for model files
        MODEL_FILES=$(find "$RESULTS_DIR" -name "*.keras" | wc -l)
        if [ $MODEL_FILES -gt 0 ]; then
            echo "✓ Found $MODEL_FILES model file(s)"
            find "$RESULTS_DIR" -name "*.keras" -exec du -h {} \;
        else
            echo "⚠ No model files found in results"
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
            echo "All data properly saved and validated in shared storage"
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