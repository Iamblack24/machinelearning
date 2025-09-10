"""
Improved logging configuration with rotation and performance monitoring.
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_optimized_logging(log_level=logging.INFO, max_file_size_mb=10, backup_count=5):
    """
    Set up optimized logging with rotation to prevent large log files.
    
    Args:
        log_level: Logging level (default: INFO)
        max_file_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup files to keep
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup rotating file handler
    log_file = os.path.join(log_dir, 'optimized_learning.log')
    max_bytes = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
    
    rotating_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    # Setup console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    rotating_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our optimized handlers
    root_logger.addHandler(rotating_handler)
    root_logger.addHandler(console_handler)
    
    # Log the setup
    logging.info(f"Optimized logging configured: {log_file} (max {max_file_size_mb}MB, {backup_count} backups)")
    
    return log_file


def cleanup_old_logs(log_dir=None, days_to_keep=7):
    """
    Clean up log files older than specified days.
    
    Args:
        log_dir: Directory containing log files (default: project logs directory)
        days_to_keep: Number of days to keep log files
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    
    if not os.path.exists(log_dir):
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    cleaned_files = 0
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log') or filename.endswith('.log.1'):
            filepath = os.path.join(log_dir, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    cleaned_files += 1
                except OSError:
                    pass  # Ignore if file cannot be removed
    
    if cleaned_files > 0:
        logging.info(f"Cleaned up {cleaned_files} old log files")


def get_log_statistics(log_dir=None):
    """
    Get statistics about log files for monitoring.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with log statistics
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    
    stats = {
        'total_log_files': 0,
        'total_size_mb': 0.0,
        'oldest_log': None,
        'newest_log': None,
        'needs_cleanup': False
    }
    
    if not os.path.exists(log_dir):
        return stats
    
    log_files = []
    total_size = 0
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log') or 'log' in filename:
            filepath = os.path.join(log_dir, filename)
            try:
                file_stat = os.stat(filepath)
                log_files.append((filename, file_stat.st_mtime, file_stat.st_size))
                total_size += file_stat.st_size
            except OSError:
                continue
    
    if log_files:
        log_files.sort(key=lambda x: x[1])  # Sort by modification time
        
        stats['total_log_files'] = len(log_files)
        stats['total_size_mb'] = total_size / (1024 * 1024)
        stats['oldest_log'] = {
            'filename': log_files[0][0],
            'timestamp': datetime.fromtimestamp(log_files[0][1]).isoformat(),
            'size_mb': log_files[0][2] / (1024 * 1024)
        }
        stats['newest_log'] = {
            'filename': log_files[-1][0],
            'timestamp': datetime.fromtimestamp(log_files[-1][1]).isoformat(),
            'size_mb': log_files[-1][2] / (1024 * 1024)
        }
        
        # Check if cleanup is needed (total size > 100MB or more than 10 files)
        stats['needs_cleanup'] = stats['total_size_mb'] > 100 or stats['total_log_files'] > 10
    
    return stats