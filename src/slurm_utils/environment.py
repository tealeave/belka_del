"""
Belka Environment Management

This module provides utilities for managing SLURM environment setup,
module loading, and environment variable configuration.
"""

import os
import logging
import platform
import psutil
from typing import Dict, Optional, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class BelkaEnvironment:
    """
    Manages SLURM environment setup and validation for Belka jobs.
    
    This class centralizes environment configuration, module loading,
    and validation logic that was previously scattered across SLURM scripts.
    """
    
    def __init__(self, cluster_type: str = "cpu", job_id: Optional[str] = None):
        """
        Initialize environment manager.
        
        Args:
            cluster_type: Type of cluster ('cpu' or 'gpu')
            job_id: SLURM job ID for logging
        """
        self.cluster_type = cluster_type
        self.job_id = job_id or os.environ.get('SLURM_JOB_ID', 'unknown')
        self.project_dir = Path("/pub/ddlin/projects/belka_del")
        self.required_modules = self._get_required_modules()
        self.env_vars = self._get_environment_variables()
        
    def _get_required_modules(self) -> Dict[str, str]:
        """Get required modules for the cluster type."""
        modules = {"python": "python/3.10.2"}
        
        if self.cluster_type == "gpu":
            modules.update({
                "cuda": "cuda/11.8",
                "cudnn": "cudnn/8.6"
            })
            
        return modules
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the cluster type."""
        env_vars = {
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "OMP_NUM_THREADS": str(min(os.cpu_count() or 1, 16)),
            "NUMEXPR_MAX_THREADS": str(min(os.cpu_count() or 1, 16))
        }
        
        if self.cluster_type == "gpu":
            env_vars.update({
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_GPU_MEMORY_ALLOCATION": "cuda_malloc_async",
                "TF_NUM_INTEROP_THREADS": "2",
                "TF_NUM_INTRAOP_THREADS": "2"
            })
            
        return env_vars
    
    def setup_environment(self) -> bool:
        """
        Set up the complete environment for the job.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info(f"Setting up environment for {self.cluster_type} cluster")
            logger.info(f"Job ID: {self.job_id}")
            
            # Set environment variables
            for var, value in self.env_vars.items():
                os.environ[var] = value
                logger.debug(f"Set {var}={value}")
            
            # Change to project directory
            os.chdir(self.project_dir)
            logger.info(f"Changed to project directory: {self.project_dir}")
            
            # Validate environment
            self._validate_environment()
            
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def _validate_environment(self) -> None:
        """Validate the environment setup."""
        # Check project directory
        if not self.project_dir.exists():
            raise RuntimeError(f"Project directory not found: {self.project_dir}")
        
        # Check Python environment
        try:
            import poetry
            logger.info("✓ Poetry environment available")
        except ImportError:
            logger.warning("Poetry not available, using system Python")
        
        # Check cluster-specific requirements
        if self.cluster_type == "gpu":
            self._validate_gpu_environment()
        else:
            self._validate_cpu_environment()
    
    def _validate_gpu_environment(self) -> None:
        """Validate GPU-specific environment."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"✓ GPU detected: {len(gpus)} device(s)")
                for i, gpu in enumerate(gpus):
                    logger.info(f"  GPU {i}: {gpu.name}")
            else:
                raise RuntimeError("No GPU devices detected")
        except ImportError:
            logger.warning("TensorFlow not available for GPU validation")
    
    def _validate_cpu_environment(self) -> None:
        """Validate CPU-specific environment."""
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"✓ CPU cores: {cpu_count}")
        logger.info(f"✓ Memory: {memory_gb:.1f}GB")
        
        if cpu_count < 2:
            logger.warning("Low CPU count detected")
        if memory_gb < 8:
            logger.warning("Low memory detected")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dict containing system information
        """
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = {
            "cluster_type": self.cluster_type,
            "job_id": self.job_id,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "percent_used": (disk.used / disk.total) * 100
            },
            "environment_variables": dict(os.environ)
        }
        
        # Add GPU info if available
        if self.cluster_type == "gpu":
            try:
                import tensorflow as tf
                info["gpu_devices"] = [str(gpu) for gpu in tf.config.list_physical_devices('GPU')]
            except ImportError:
                info["gpu_devices"] = "TensorFlow not available"
        
        return info
    
    def log_system_info(self) -> None:
        """Log comprehensive system information."""
        info = self.get_system_info()
        
        logger.info("=" * 50)
        logger.info(f"BELKA {self.cluster_type.upper()} ENVIRONMENT")
        logger.info("=" * 50)
        logger.info(f"Job ID: {info['job_id']}")
        logger.info(f"Hostname: {info['hostname']}")
        logger.info(f"Platform: {info['platform']}")
        logger.info(f"CPU cores: {info['cpu_count']}")
        logger.info(f"Memory: {info['memory']['total_gb']:.1f}GB total, "
                   f"{info['memory']['available_gb']:.1f}GB available "
                   f"({info['memory']['percent_used']:.1f}% used)")
        logger.info(f"Disk: {info['disk']['total_gb']:.1f}GB total, "
                   f"{info['disk']['used_gb']:.1f}GB used "
                   f"({info['disk']['percent_used']:.1f}% used)")
        
        if self.cluster_type == "gpu" and "gpu_devices" in info:
            logger.info(f"GPU devices: {info['gpu_devices']}")
        
        logger.info("=" * 50)