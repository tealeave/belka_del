"""
Belka Job Management

This module provides utilities for managing SLURM jobs, tracking status,
and providing comprehensive reporting and logging.
"""

import os
import logging
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
import psutil
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Job status information."""
    job_id: str
    job_name: str
    start_time: datetime
    cluster_type: str
    status: str = "running"
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None


class BelkaJobManager:
    """
    Manages SLURM job execution, monitoring, and reporting.
    
    This class provides comprehensive job management capabilities
    including status tracking, resource monitoring, and reporting.
    """
    
    def __init__(self, 
                 job_name: str,
                 cluster_type: str = "cpu",
                 project_root: Optional[str] = None):
        """
        Initialize job manager.
        
        Args:
            job_name: Name of the job
            cluster_type: Type of cluster ('cpu' or 'gpu')
            project_root: Root directory for the project
        """
        self.job_name = job_name
        self.cluster_type = cluster_type
        self.project_root = Path(project_root or "/pub/ddlin/projects/belka_del")
        self.job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        self.node_name = os.environ.get('SLURMD_NODENAME', 'unknown')
        
        self.start_time = datetime.now()
        self.status = JobStatus(
            job_id=self.job_id,
            job_name=job_name,
            start_time=self.start_time,
            cluster_type=cluster_type
        )
        
        # Resource monitoring
        self.initial_resources = self._get_resource_snapshot()
        self.resource_history: List[Dict[str, Any]] = []
        
    def log_job_start(self) -> None:
        """Log job start information."""
        logger.info("=" * 60)
        logger.info(f"BELKA {self.cluster_type.upper()} JOB STARTED")
        logger.info("=" * 60)
        logger.info(f"Job Name: {self.job_name}")
        logger.info(f"Job ID: {self.job_id}")
        logger.info(f"Node: {self.node_name}")
        logger.info(f"Cluster Type: {self.cluster_type}")
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Project Root: {self.project_root}")
        logger.info("=" * 60)
        
        # Log initial resource state
        self._log_resources("Job Start")
    
    def log_job_end(self, exit_code: int = 0, error_message: Optional[str] = None) -> None:
        """
        Log job completion information.
        
        Args:
            exit_code: Job exit code (0 = success)
            error_message: Optional error message
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Update status
        self.status.end_time = end_time
        self.status.duration_seconds = duration
        self.status.exit_code = exit_code
        self.status.error_message = error_message
        self.status.status = "completed" if exit_code == 0 else "failed"
        
        # Log completion
        logger.info("=" * 60)
        if exit_code == 0:
            logger.info(f"BELKA {self.cluster_type.upper()} JOB COMPLETED SUCCESSFULLY")
        else:
            logger.error(f"BELKA {self.cluster_type.upper()} JOB FAILED")
        logger.info("=" * 60)
        logger.info(f"Job Name: {self.job_name}")
        logger.info(f"Job ID: {self.job_id}")
        logger.info(f"Exit Code: {exit_code}")
        logger.info(f"Duration: {self._format_duration(duration)}")
        
        if error_message:
            logger.error(f"Error: {error_message}")
        
        # Log final resource state
        self._log_resources("Job End")
        
        # Log resource summary
        self._log_resource_summary()
        
        logger.info("=" * 60)
    
    def monitor_resources(self, context: str = "") -> Dict[str, Any]:
        """
        Monitor and log current resource usage.
        
        Args:
            context: Context description for logging
            
        Returns:
            Dict containing resource information
        """
        resources = self._get_resource_snapshot()
        resources["timestamp"] = datetime.now()
        resources["context"] = context
        
        # Add to history
        self.resource_history.append(resources)
        
        # Log if context provided
        if context:
            self._log_resources(context, resources)
        
        return resources
    
    def _get_resource_snapshot(self) -> Dict[str, Any]:
        """Get current resource snapshot."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.project_root))
        cpu_percent = psutil.cpu_percent(interval=1)
        
        resources = {
            "memory": {
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent_used": (disk.used / disk.total) * 100
            },
            "cpu": {
                "percent_used": cpu_percent,
                "core_count": os.cpu_count() or 1
            }
        }
        
        # Add GPU info if available
        if self.cluster_type == "gpu":
            resources["gpu"] = self._get_gpu_info()
        
        return resources
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return {
                "devices_available": len(gpus),
                "device_names": [gpu.name for gpu in gpus]
            }
        except ImportError:
            return {"error": "TensorFlow not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _log_resources(self, context: str, resources: Optional[Dict[str, Any]] = None) -> None:
        """Log resource information."""
        if resources is None:
            resources = self._get_resource_snapshot()
        
        mem = resources["memory"]
        disk = resources["disk"]
        cpu = resources["cpu"]
        
        logger.info(f"System Resources [{context}]:")
        logger.info(f"  Memory: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB "
                   f"({mem['percent_used']:.1f}% used)")
        logger.info(f"  CPU: {cpu['percent_used']:.1f}% ({cpu['core_count']} cores)")
        logger.info(f"  Disk: {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB "
                   f"({disk['percent_used']:.1f}% used)")
        
        if "gpu" in resources:
            gpu = resources["gpu"]
            if "error" in gpu:
                logger.warning(f"  GPU: {gpu['error']}")
            else:
                logger.info(f"  GPU: {gpu['devices_available']} device(s) available")
    
    def _log_resource_summary(self) -> None:
        """Log summary of resource usage throughout job."""
        if len(self.resource_history) < 2:
            return
        
        logger.info("Resource Usage Summary:")
        
        # Memory usage over time
        memory_usage = [r["memory"]["percent_used"] for r in self.resource_history]
        logger.info(f"  Memory Peak: {max(memory_usage):.1f}%")
        logger.info(f"  Memory Average: {sum(memory_usage)/len(memory_usage):.1f}%")
        
        # CPU usage over time
        cpu_usage = [r["cpu"]["percent_used"] for r in self.resource_history]
        logger.info(f"  CPU Peak: {max(cpu_usage):.1f}%")
        logger.info(f"  CPU Average: {sum(cpu_usage)/len(cpu_usage):.1f}%")
        
        # Disk usage change
        initial_disk = self.initial_resources["disk"]["percent_used"]
        final_disk = self.resource_history[-1]["disk"]["percent_used"]
        disk_change = final_disk - initial_disk
        logger.info(f"  Disk Usage Change: {disk_change:+.1f}%")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def check_slurm_status(self) -> Dict[str, Any]:
        """
        Check SLURM job status using squeue.
        
        Returns:
            Dict containing SLURM status information
        """
        try:
            result = subprocess.run(
                ['squeue', '-j', self.job_id, '-h', '-o', '%i,%j,%t,%M,%N'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                fields = result.stdout.strip().split(',')
                return {
                    "job_id": fields[0],
                    "job_name": fields[1],
                    "state": fields[2],
                    "time": fields[3],
                    "nodes": fields[4]
                }
            else:
                return {"error": "Job not found in queue"}
                
        except subprocess.TimeoutExpired:
            return {"error": "SLURM command timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_job_summary(self) -> Dict[str, Any]:
        """Get comprehensive job summary."""
        summary = asdict(self.status)
        summary.update({
            "node_name": self.node_name,
            "project_root": str(self.project_root),
            "resource_snapshots": len(self.resource_history),
            "current_resources": self._get_resource_snapshot(),
            "slurm_status": self.check_slurm_status()
        })
        
        return summary
    
    def save_job_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Save detailed job report to file.
        
        Args:
            output_path: Optional output path for report
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = (self.project_root / "logs" / 
                          f"job_report_{self.job_id}_{int(time.time())}.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Prepare data for JSON serialization
        summary = self.get_job_summary()
        
        # Convert datetime objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Convert all datetime objects
        def convert_datetimes(data):
            if isinstance(data, dict):
                return {k: convert_datetimes(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_datetimes(item) for item in data]
            elif isinstance(data, datetime):
                return data.isoformat()
            else:
                return data
        
        summary = convert_datetimes(summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Job report saved to: {output_path}")
        return output_path