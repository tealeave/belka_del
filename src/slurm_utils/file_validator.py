"""
Belka File Validation

This module provides utilities for validating files and datasets
with comprehensive error reporting and logging.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass
import time


logger = logging.getLogger(__name__)


@dataclass
class FileValidationResult:
    """Result of file validation."""
    path: str
    exists: bool
    is_valid: bool
    file_type: str
    size_bytes: int
    size_human: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BelkaFileValidator:
    """
    Validates files and datasets for Belka jobs.
    
    This class provides comprehensive file validation with detailed
    error reporting and logging capabilities.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize file validator.
        
        Args:
            project_root: Root directory for the project
        """
        self.project_root = Path(project_root or "/pub/ddlin/projects/belka_del")
        self.required_input_files = [
            "data/raw/train.parquet",
            "data/raw/test.parquet", 
            "data/raw/DNA_Labeled_Data.csv"
        ]
        self.expected_output_files = [
            "data/raw/belka.parquet",  # Directory
            "data/raw/vocab.txt"
        ]
        
    def validate_input_files(self) -> Dict[str, FileValidationResult]:
        """
        Validate all required input files.
        
        Returns:
            Dict mapping file paths to validation results
        """
        logger.info("Validating required input files...")
        
        results = {}
        all_valid = True
        
        for file_path in self.required_input_files:
            full_path = self.project_root / file_path
            result = self._validate_single_file(full_path)
            results[file_path] = result
            
            if result.is_valid:
                logger.info(f"✓ {file_path}: {result.size_human}")
            else:
                logger.error(f"✗ {file_path}: {result.error_message}")
                all_valid = False
        
        if all_valid:
            logger.info("✓ All required input files are valid")
        else:
            logger.error("✗ Some required input files are missing or invalid")
        
        return results
    
    def validate_output_files(self) -> Dict[str, FileValidationResult]:
        """
        Validate expected output files.
        
        Returns:
            Dict mapping file paths to validation results
        """
        logger.info("Validating expected output files...")
        
        results = {}
        all_valid = True
        
        for file_path in self.expected_output_files:
            full_path = self.project_root / file_path
            result = self._validate_single_file(full_path)
            results[file_path] = result
            
            if result.is_valid:
                logger.info(f"✓ {file_path}: {result.size_human}")
            else:
                logger.error(f"✗ {file_path}: {result.error_message}")
                all_valid = False
        
        if all_valid:
            logger.info("✓ All expected output files are valid")
        else:
            logger.error("✗ Some expected output files are missing or invalid")
        
        return results
    
    def _validate_single_file(self, path: Path) -> FileValidationResult:
        """
        Validate a single file or directory.
        
        Args:
            path: Path to validate
            
        Returns:
            FileValidationResult object
        """
        try:
            path = Path(path)
            
            # Check if path exists
            if not path.exists():
                return FileValidationResult(
                    path=str(path),
                    exists=False,
                    is_valid=False,
                    file_type="missing",
                    size_bytes=0,
                    size_human="0B",
                    error_message="File does not exist"
                )
            
            # Get basic info
            file_type = self._get_file_type(path)
            size_bytes = self._get_path_size(path)
            size_human = self._format_size(size_bytes)
            
            # Validate based on file type
            is_valid, error_message, metadata = self._validate_by_type(path, file_type)
            
            return FileValidationResult(
                path=str(path),
                exists=True,
                is_valid=is_valid,
                file_type=file_type,
                size_bytes=size_bytes,
                size_human=size_human,
                error_message=error_message,
                metadata=metadata
            )
            
        except Exception as e:
            return FileValidationResult(
                path=str(path),
                exists=False,
                is_valid=False,
                file_type="error",
                size_bytes=0,
                size_human="0B",
                error_message=f"Validation error: {str(e)}"
            )
    
    def _get_file_type(self, path: Path) -> str:
        """Determine file type."""
        if path.is_dir():
            return "directory"
        elif path.suffix.lower() == '.parquet':
            return "parquet"
        elif path.suffix.lower() == '.csv':
            return "csv"
        elif path.suffix.lower() == '.txt':
            return "text"
        elif path.suffix.lower() == '.tfr':
            return "tfrecord"
        else:
            return "unknown"
    
    def _get_path_size(self, path: Path) -> int:
        """Get size of file or directory."""
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = Path(dirpath) / filename
                        try:
                            total_size += filepath.stat().st_size
                        except (OSError, IOError):
                            pass
                return total_size
            else:
                return 0
        except Exception:
            return 0
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    def _validate_by_type(self, path: Path, file_type: str) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate file based on its type.
        
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        try:
            if file_type == "parquet":
                return self._validate_parquet(path)
            elif file_type == "csv":
                return self._validate_csv(path)
            elif file_type == "text":
                return self._validate_text(path)
            elif file_type == "directory":
                return self._validate_directory(path)
            elif file_type == "tfrecord":
                return self._validate_tfrecord(path)
            else:
                return True, None, None
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def _validate_parquet(self, path: Path) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate parquet file."""
        try:
            # Try to read parquet metadata
            parquet_file = pq.ParquetFile(path)
            metadata = {
                "num_rows": parquet_file.metadata.num_rows,
                "num_columns": parquet_file.schema.names.__len__(),
                "columns": parquet_file.schema.names,
                "file_size": path.stat().st_size
            }
            
            if metadata["num_rows"] == 0:
                return False, "Parquet file is empty", metadata
            
            return True, None, metadata
            
        except Exception as e:
            return False, f"Invalid parquet file: {str(e)}", None
    
    def _validate_csv(self, path: Path) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate CSV file."""
        try:
            # Try to read CSV header
            df_sample = pd.read_csv(path, nrows=5)
            metadata = {
                "num_columns": len(df_sample.columns),
                "columns": df_sample.columns.tolist(),
                "file_size": path.stat().st_size
            }
            
            if len(df_sample) == 0:
                return False, "CSV file is empty", metadata
            
            return True, None, metadata
            
        except Exception as e:
            return False, f"Invalid CSV file: {str(e)}", None
    
    def _validate_text(self, path: Path) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate text file."""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            metadata = {
                "num_lines": len(lines),
                "file_size": path.stat().st_size
            }
            
            if len(lines) == 0:
                return False, "Text file is empty", metadata
            
            # For vocab.txt, check if it has reasonable content
            if path.name == "vocab.txt":
                if len(lines) < 10:
                    return False, f"Vocab file has too few tokens: {len(lines)}", metadata
                
                # Check for common SMILES tokens
                content = ''.join(lines).lower()
                if not any(token in content for token in ['c', 'n', 'o', '(', ')', '[', ']']):
                    return False, "Vocab file doesn't contain expected SMILES tokens", metadata
            
            return True, None, metadata
            
        except Exception as e:
            return False, f"Invalid text file: {str(e)}", None
    
    def _validate_directory(self, path: Path) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate directory (e.g., parquet directory)."""
        try:
            files = list(path.glob("*"))
            
            if len(files) == 0:
                return False, "Directory is empty", None
            
            metadata = {
                "num_files": len(files),
                "file_types": [f.suffix for f in files if f.is_file()],
                "total_size": self._get_path_size(path)
            }
            
            # Special validation for parquet directories
            if path.name == "belka.parquet":
                parquet_files = list(path.glob("*.parquet"))
                if len(parquet_files) == 0:
                    return False, "No parquet files found in directory", metadata
                
                # Try to read one parquet file to validate format
                try:
                    pq.ParquetFile(parquet_files[0])
                    metadata["num_parquet_files"] = len(parquet_files)
                except Exception as e:
                    return False, f"Invalid parquet files in directory: {str(e)}", metadata
            
            return True, None, metadata
            
        except Exception as e:
            return False, f"Invalid directory: {str(e)}", None
    
    def _validate_tfrecord(self, path: Path) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate TFRecord file."""
        try:
            # Basic size check
            size = path.stat().st_size
            if size == 0:
                return False, "TFRecord file is empty", None
            
            metadata = {
                "file_size": size
            }
            
            # Try to read first record to validate format
            try:
                import tensorflow as tf
                dataset = tf.data.TFRecordDataset(str(path))
                first_record = next(iter(dataset.take(1)))
                metadata["has_valid_records"] = True
            except Exception:
                metadata["has_valid_records"] = False
            
            return True, None, metadata
            
        except Exception as e:
            return False, f"Invalid TFRecord file: {str(e)}", None
    
    def generate_validation_report(self, results: Dict[str, FileValidationResult]) -> str:
        """
        Generate a detailed validation report.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("BELKA FILE VALIDATION REPORT")
        report.append("=" * 60)
        
        valid_count = sum(1 for r in results.values() if r.is_valid)
        total_count = len(results)
        
        report.append(f"Overall Status: {valid_count}/{total_count} files valid")
        report.append("")
        
        for file_path, result in results.items():
            status = "✓" if result.is_valid else "✗"
            report.append(f"{status} {file_path}")
            report.append(f"  Type: {result.file_type}")
            report.append(f"  Size: {result.size_human}")
            
            if result.error_message:
                report.append(f"  Error: {result.error_message}")
            
            if result.metadata:
                for key, value in result.metadata.items():
                    report.append(f"  {key}: {value}")
            
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def log_validation_results(self, results: Dict[str, FileValidationResult]) -> None:
        """Log validation results."""
        report = self.generate_validation_report(results)
        logger.info(report)