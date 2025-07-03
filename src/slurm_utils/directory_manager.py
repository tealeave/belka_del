"""
Belka Directory Management

This module provides utilities for managing directories and file operations
in a safe, robust manner with proper error handling and logging.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import stat
import time


logger = logging.getLogger(__name__)


class BelkaDirectoryManager:
    """
    Manages directory operations for Belka jobs.
    
    This class provides safe, robust directory operations with proper
    error handling, logging, and validation.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize directory manager.
        
        Args:
            project_root: Root directory for the project
        """
        self.project_root = Path(project_root or "/pub/ddlin/projects/belka_del")
        self.standard_dirs = {
            "data": self.project_root / "data",
            "data_raw": self.project_root / "data" / "raw",
            "data_processed": self.project_root / "data" / "processed",
            "logs": self.project_root / "logs",
            "models": self.project_root / "models",
            "checkpoints": self.project_root / "checkpoints",
            "results": self.project_root / "results"
        }
        
    def create_standard_directories(self) -> bool:
        """
        Create all standard directories for the project.
        
        Returns:
            bool: True if all directories created successfully
        """
        try:
            logger.info("Creating standard project directories...")
            
            for name, path in self.standard_dirs.items():
                if self.create_directory(path):
                    logger.debug(f"✓ {name}: {path}")
                else:
                    logger.error(f"✗ Failed to create {name}: {path}")
                    return False
            
            logger.info("✓ All standard directories created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create standard directories: {e}")
            return False
    
    def create_directory(self, path: Path, mode: int = 0o755) -> bool:
        """
        Create a directory with proper error handling.
        
        Args:
            path: Directory path to create
            mode: Directory permissions (default: 0o755)
            
        Returns:
            bool: True if directory created or already exists
        """
        try:
            path = Path(path)
            
            if path.exists():
                if path.is_dir():
                    logger.debug(f"Directory already exists: {path}")
                    return True
                else:
                    logger.error(f"Path exists but is not a directory: {path}")
                    return False
            
            # Create directory with parents
            path.mkdir(parents=True, exist_ok=True, mode=mode)
            logger.debug(f"Created directory: {path}")
            return True
            
        except PermissionError:
            logger.error(f"Permission denied creating directory: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    def safe_remove(self, path: Path, dry_run: bool = False) -> bool:
        """
        Safely remove a file or directory with validation.
        
        Args:
            path: Path to remove
            dry_run: If True, only log what would be removed
            
        Returns:
            bool: True if removal successful or dry run
        """
        try:
            path = Path(path)
            
            if not path.exists():
                logger.debug(f"Path does not exist: {path}")
                return True
            
            # Validate path is within project directory for safety
            if not self._is_safe_path(path):
                logger.error(f"Unsafe path for removal: {path}")
                return False
            
            # Log what will be removed
            if path.is_file():
                size = path.stat().st_size
                logger.info(f"{'Would remove' if dry_run else 'Removing'} file: {path} ({size} bytes)")
            elif path.is_dir():
                size = self._get_directory_size(path)
                logger.info(f"{'Would remove' if dry_run else 'Removing'} directory: {path} ({size} bytes)")
            
            if dry_run:
                return True
            
            # Perform removal
            if path.is_file():
                path.unlink()
                logger.debug(f"Removed file: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                logger.debug(f"Removed directory: {path}")
            
            return True
            
        except PermissionError:
            logger.error(f"Permission denied removing: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to remove {path}: {e}")
            return False
    
    def clean_previous_outputs(self, dry_run: bool = False, force: bool = False) -> bool:
        """
        Clean previous output files as done in SLURM scripts.
        
        Args:
            dry_run: If True, only log what would be cleaned
            force: If True, bypass safety check and delete existing files
            
        Returns:
            bool: True if cleaning successful
        """
        try:
            # Files to clean (from SLURM scripts)
            files_to_clean = [
                self.project_root / "data" / "raw" / "belka.parquet",
                self.project_root / "data" / "raw" / "vocab.txt"
            ]
            
            # Check if any files exist
            existing_files = []
            for file_path in files_to_clean:
                if file_path.exists():
                    size = file_path.stat().st_size
                    existing_files.append((file_path, size))
            
            if existing_files and not force:
                # Error out if files exist and force is not enabled
                logger.error("✗ Cannot proceed: Existing output files detected!")
                logger.error("Found files that would be overwritten:")
                
                for file_path, size in existing_files:
                    # Human readable size
                    size_human = self._format_size(size)
                    file_stat = file_path.stat()
                    modified_time = time.ctime(file_stat.st_mtime)
                    logger.error(f"  • {file_path.name} ({size_human}, modified: {modified_time})")
                
                logger.error("Use --force-clean flag to overwrite existing files.")
                return False
            
            if existing_files:
                logger.info("Cleaning previous processed data...")
                if force:
                    logger.info("⚠️  FORCE mode enabled - deleting existing files")
            else:
                logger.info("No previous output files found to clean")
                return True
            
            success = True
            for file_path in files_to_clean:
                if not self.safe_remove(file_path, dry_run=dry_run):
                    success = False
            
            if success:
                logger.info("✓ Previous data cleaned successfully")
            else:
                logger.error("✗ Some files failed to clean")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to clean previous outputs: {e}")
            return False
    
    def _is_safe_path(self, path: Path) -> bool:
        """
        Check if a path is safe for removal (within project directory).
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if path is safe
        """
        try:
            path = Path(path).resolve()
            project_root = self.project_root.resolve()
            
            # Check if path is within project directory
            try:
                path.relative_to(project_root)
                return True
            except ValueError:
                return False
                
        except Exception:
            return False
    
    def _get_directory_size(self, path: Path) -> int:
        """
        Get total size of a directory in bytes.
        
        Args:
            path: Directory path
            
        Returns:
            int: Size in bytes
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    try:
                        total_size += filepath.stat().st_size
                    except (OSError, IOError):
                        pass
            return total_size
        except Exception:
            return 0
    
    def _format_size(self, size_bytes: int) -> str:
        """
        Format size in bytes to human readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            str: Human readable size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    def get_directory_info(self, path: Path) -> Dict[str, Any]:
        """
        Get comprehensive information about a directory.
        
        Args:
            path: Directory path
            
        Returns:
            Dict containing directory information
        """
        path = Path(path)
        
        info = {
            "path": str(path),
            "exists": path.exists(),
            "is_directory": path.is_dir() if path.exists() else False,
            "is_file": path.is_file() if path.exists() else False,
            "size_bytes": 0,
            "size_human": "0B",
            "file_count": 0,
            "permissions": None,
            "last_modified": None
        }
        
        if path.exists():
            try:
                stat_info = path.stat()
                info["permissions"] = oct(stat_info.st_mode)[-3:]
                info["last_modified"] = time.ctime(stat_info.st_mtime)
                
                if path.is_dir():
                    info["size_bytes"] = self._get_directory_size(path)
                    info["file_count"] = len(list(path.rglob("*")))
                else:
                    info["size_bytes"] = stat_info.st_size
                    info["file_count"] = 1
                
                # Human readable size
                size_bytes = info["size_bytes"]
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if size_bytes < 1024.0:
                        info["size_human"] = f"{size_bytes:.1f}{unit}"
                        break
                    size_bytes /= 1024.0
                
            except Exception as e:
                logger.debug(f"Error getting info for {path}: {e}")
        
        return info
    
    def log_directory_info(self, directories: List[Path]) -> None:
        """
        Log information about multiple directories.
        
        Args:
            directories: List of directory paths to log
        """
        logger.info("Directory Information:")
        logger.info("-" * 50)
        
        for directory in directories:
            info = self.get_directory_info(directory)
            
            status = "✓" if info["exists"] else "✗"
            type_str = "DIR" if info["is_directory"] else "FILE" if info["is_file"] else "N/A"
            
            logger.info(f"{status} {directory.name:<20} [{type_str}] {info['size_human']:<10} "
                       f"({info['file_count']} items)")
        
        logger.info("-" * 50)