"""
Configuration loader utility for YAML config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Loader for configuration files from YAML."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize config loader.
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_absolute():
            # Make path relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / self.config_dir
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file.
        
        Args:
            config_name: Name of config file without extension
        
        Returns:
            Dictionary with configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
    
    def load_all(self) -> Dict[str, Any]:
        """Load all available configuration files.
        
        Returns:
            Dictionary with all configurations
        """
        configs = {}
        
        if not self.config_dir.exists():
            return configs
            
        for yaml_file in self.config_dir.glob("*.yaml"):
            config_name = yaml_file.stem
            try:
                configs[config_name] = self.load(config_name)
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load {config_name}: {e}")
        
        return configs
    
    def get_path(self, *path_parts: str) -> Path:
        """Get absolute path from config relative path.
        
        Args:
            *path_parts: Path components
            
        Returns:
            Absolute path
        """
        project_root = Path(__file__).parent.parent.parent
        return project_root / Path(*path_parts)


# Global instance for easy access
config_loader = ConfigLoader()