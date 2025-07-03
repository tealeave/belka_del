#!/usr/bin/env python3
"""
Belka SLURM Utilities CLI

Command-line interface for SLURM job utilities including environment setup,
file validation, directory management, and job monitoring.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.slurm_utils import (
    BelkaEnvironment,
    BelkaDirectoryManager,
    BelkaFileValidator,
    BelkaJobManager
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_setup_environment(args):
    """Setup SLURM environment."""
    env = BelkaEnvironment(cluster_type=args.cluster_type)
    
    if env.setup_environment():
        env.log_system_info()
        print("✓ Environment setup completed successfully")
        return 0
    else:
        print("✗ Environment setup failed")
        return 1


def cmd_create_directories(args):
    """Create standard project directories."""
    dir_manager = BelkaDirectoryManager()
    
    if dir_manager.create_standard_directories():
        print("✓ Standard directories created successfully")
        if args.verbose:
            dir_manager.log_directory_info(list(dir_manager.standard_dirs.values()))
        return 0
    else:
        print("✗ Failed to create standard directories")
        return 1


def cmd_clean_outputs(args):
    """Clean previous output files."""
    dir_manager = BelkaDirectoryManager()
    
    if dir_manager.clean_previous_outputs(dry_run=args.dry_run, force=args.force_clean):
        if args.dry_run:
            print("✓ Dry run completed - no files were actually removed")
        else:
            print("✓ Previous outputs cleaned successfully")
        return 0
    else:
        print("✗ Failed to clean previous outputs")
        return 1


def cmd_validate_inputs(args):
    """Validate required input files."""
    validator = BelkaFileValidator()
    
    results = validator.validate_input_files()
    
    if args.format == "json":
        # Convert results to JSON-serializable format
        json_results = {}
        for path, result in results.items():
            json_results[path] = {
                "path": result.path,
                "exists": result.exists,
                "is_valid": result.is_valid,
                "file_type": result.file_type,
                "size_bytes": result.size_bytes,
                "size_human": result.size_human,
                "error_message": result.error_message,
                "metadata": result.metadata
            }
        print(json.dumps(json_results, indent=2))
    else:
        validator.log_validation_results(results)
    
    # Return 0 if all files are valid, 1 otherwise
    all_valid = all(result.is_valid for result in results.values())
    return 0 if all_valid else 1


def cmd_validate_outputs(args):
    """Validate expected output files."""
    validator = BelkaFileValidator()
    
    results = validator.validate_output_files()
    
    if args.format == "json":
        # Convert results to JSON-serializable format
        json_results = {}
        for path, result in results.items():
            json_results[path] = {
                "path": result.path,
                "exists": result.exists,
                "is_valid": result.is_valid,
                "file_type": result.file_type,
                "size_bytes": result.size_bytes,
                "size_human": result.size_human,
                "error_message": result.error_message,
                "metadata": result.metadata
            }
        print(json.dumps(json_results, indent=2))
    else:
        validator.log_validation_results(results)
    
    # Return 0 if all files are valid, 1 otherwise
    all_valid = all(result.is_valid for result in results.values())
    return 0 if all_valid else 1


def cmd_monitor_resources(args):
    """Monitor system resources."""
    job_manager = BelkaJobManager(
        job_name=args.job_name or "monitor",
        cluster_type=args.cluster_type
    )
    
    resources = job_manager.monitor_resources(context=args.context or "Manual Check")
    
    if args.format == "json":
        # Convert datetime to string for JSON serialization
        resources["timestamp"] = resources["timestamp"].isoformat()
        print(json.dumps(resources, indent=2))
    
    return 0


def cmd_job_start(args):
    """Log job start information."""
    job_manager = BelkaJobManager(
        job_name=args.job_name,
        cluster_type=args.cluster_type
    )
    
    job_manager.log_job_start()
    
    # Save job manager state for later use
    state_file = Path(f"/tmp/belka_job_{job_manager.job_id}.json")
    job_summary = job_manager.get_job_summary()
    
    # Convert datetime objects for JSON serialization
    def convert_datetimes(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return obj
    
    # Convert datetime in job_summary
    if 'start_time' in job_summary:
        job_summary['start_time'] = convert_datetimes(job_summary['start_time'])
    
    with open(state_file, 'w') as f:
        json.dump(job_summary, f, indent=2, default=convert_datetimes)
    
    print(f"Job started. State saved to: {state_file}")
    return 0


def cmd_job_end(args):
    """Log job end information."""
    job_manager = BelkaJobManager(
        job_name=args.job_name,
        cluster_type=args.cluster_type
    )
    
    job_manager.log_job_end(
        exit_code=args.exit_code,
        error_message=args.error_message
    )
    
    # Save job report
    if args.save_report:
        report_path = job_manager.save_job_report()
        print(f"Job report saved to: {report_path}")
    
    return 0


def cmd_system_info(args):
    """Display comprehensive system information."""
    env = BelkaEnvironment(cluster_type=args.cluster_type)
    info = env.get_system_info()
    
    if args.format == "json":
        print(json.dumps(info, indent=2, default=str))
    else:
        env.log_system_info()
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Belka SLURM Utilities CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--cluster-type",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Cluster type"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Environment setup
    env_parser = subparsers.add_parser("setup-env", help="Setup SLURM environment")
    env_parser.set_defaults(func=cmd_setup_environment)
    
    # Directory management
    dirs_parser = subparsers.add_parser("create-dirs", help="Create standard directories")
    dirs_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    dirs_parser.set_defaults(func=cmd_create_directories)
    
    # Cleanup
    clean_parser = subparsers.add_parser("clean", help="Clean previous outputs")
    clean_parser.add_argument("--dry-run", action="store_true", help="Dry run (don't actually remove)")
    clean_parser.add_argument("--force-clean", action="store_true", help="Force deletion of existing files")
    clean_parser.set_defaults(func=cmd_clean_outputs)
    
    # Input validation
    validate_inputs_parser = subparsers.add_parser("validate-inputs", help="Validate input files")
    validate_inputs_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    validate_inputs_parser.set_defaults(func=cmd_validate_inputs)
    
    # Output validation
    validate_outputs_parser = subparsers.add_parser("validate-outputs", help="Validate output files")
    validate_outputs_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    validate_outputs_parser.set_defaults(func=cmd_validate_outputs)
    
    # Resource monitoring
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system resources")
    monitor_parser.add_argument("--context", help="Context description")
    monitor_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    monitor_parser.add_argument("--job-name", help="Job name for monitoring")
    monitor_parser.set_defaults(func=cmd_monitor_resources)
    
    # Job management
    job_start_parser = subparsers.add_parser("job-start", help="Log job start")
    job_start_parser.add_argument("job_name", help="Job name")
    job_start_parser.set_defaults(func=cmd_job_start)
    
    job_end_parser = subparsers.add_parser("job-end", help="Log job end")
    job_end_parser.add_argument("job_name", help="Job name")
    job_end_parser.add_argument("--exit-code", type=int, default=0, help="Job exit code")
    job_end_parser.add_argument("--error-message", help="Error message if job failed")
    job_end_parser.add_argument("--save-report", action="store_true", help="Save detailed job report")
    job_end_parser.set_defaults(func=cmd_job_end)
    
    # System info
    info_parser = subparsers.add_parser("system-info", help="Display system information")
    info_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    info_parser.set_defaults(func=cmd_system_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        return args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())