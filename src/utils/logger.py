"""
Logging utility for the exoplanet detection pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "exoplanet_hunter",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """Set up a logger with console and file output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console_output: Whether to output to console
        file_output: Whether to output to file
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    if not log_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        log_path = project_root / log_path
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name, defaults to 'exoplanet_hunter'
        
    Returns:
        Logger instance
    """
    if name is None:
        name = "exoplanet_hunter"
    return logging.getLogger(name)


# Global logger instance
logger = setup_logger()