"""
Data loaders and dataset classes for PyTorch training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import lightkurve as lk
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader
from .catalog import CatalogManager

logger = get_logger(__name__)


class LightCurveDataset(Dataset):
    """PyTorch Dataset for light curve data."""
    
    def __init__(
        self,
        light_curves: List[np.ndarray],
        labels: List[int],
        stellar_features: Optional[List[np.ndarray]] = None,
        target_ids: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        augment: bool = False
    ):
        """Initialize dataset.
        
        Args:
            light_curves: List of light curve arrays
            labels: List of labels (0 or 1)
            stellar_features: Optional stellar parameter features
            target_ids: Optional target identifiers
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        self.light_curves = light_curves
        self.labels = labels
        self.stellar_features = stellar_features
        self.target_ids = target_ids or [f"target_{i}" for i in range(len(light_curves))]
        self.transform = transform
        self.augment = augment
        
        assert len(self.light_curves) == len(self.labels), "Length mismatch between light curves and labels"
        
        if self.stellar_features is not None:
            assert len(self.stellar_features) == len(self.labels), "Length mismatch between stellar features and labels"
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.light_curves)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with tensors
        """
        # Get light curve
        light_curve = self.light_curves[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment and np.random.random() < 0.5:
            light_curve = self._augment_light_curve(light_curve)
        
        # Apply transform
        if self.transform:
            light_curve = self.transform(light_curve)
        
        # Convert to tensors
        item = {
            'light_curve': torch.FloatTensor(light_curve),
            'label': torch.LongTensor([label]),
            'target_id': self.target_ids[idx]
        }
        
        # Add stellar features if available
        if self.stellar_features is not None:
            stellar_feat = self.stellar_features[idx]
            item['stellar_features'] = torch.FloatTensor(stellar_feat)
        
        return item
    
    def _augment_light_curve(self, light_curve: np.ndarray) -> np.ndarray:
        """Apply data augmentation to light curve.
        
        Args:
            light_curve: Input light curve
            
        Returns:
            Augmented light curve
        """
        config = config_loader.load('config')
        augment_config = config['training']['augmentation']
        
        # Add Gaussian noise
        if augment_config.get('noise_level', 0) > 0:
            noise = np.random.normal(0, augment_config['noise_level'], light_curve.shape)
            light_curve = light_curve + noise
        
        # Time shift (circular shift)
        if augment_config.get('time_shift', False):
            shift = np.random.randint(-len(light_curve)//10, len(light_curve)//10)
            light_curve = np.roll(light_curve, shift)
        
        # Amplitude scaling
        if augment_config.get('amplitude_scale', 0) > 0:
            scale = 1 + np.random.uniform(-augment_config['amplitude_scale'], 
                                         augment_config['amplitude_scale'])
            light_curve = light_curve * scale
        
        return light_curve


class ExoplanetDataLoader:
    """Data loader for exoplanet detection pipeline."""
    
    def __init__(self):
        """Initialize data loader."""
        self.config = config_loader.load('config')
        self.catalog_manager = CatalogManager()
        self.scalers = {}
        
        self.raw_dir = config_loader.get_path(self.config['data']['raw_dir'])
        self.processed_dir = config_loader.get_path(self.config['data']['processed_dir'])
    
    def load_light_curve_from_fits(
        self, 
        fits_path: Union[str, Path]
    ) -> Optional[np.ndarray]:
        """Load light curve from FITS file.
        
        Args:
            fits_path: Path to FITS file
            
        Returns:
            Light curve array or None if failed
        """
        try:
            lc = lk.read(fits_path)
            
            # Remove NaN values
            mask = np.isfinite(lc.flux.value)
            time = lc.time.value[mask]
            flux = lc.flux.value[mask]
            
            if len(flux) < 100:  # Minimum length check
                logger.warning(f"Light curve too short: {len(flux)} points")
                return None
            
            return flux
            
        except Exception as e:
            logger.error(f"Error loading FITS file {fits_path}: {e}")
            return None
    
    def normalize_light_curve(
        self, 
        flux: np.ndarray, 
        method: str = 'robust'
    ) -> np.ndarray:
        """Normalize light curve.
        
        Args:
            flux: Flux array
            method: Normalization method ('standard', 'robust', 'minmax')
            
        Returns:
            Normalized flux
        """
        if method == 'standard':
            # Z-score normalization
            mean_flux = np.mean(flux)
            std_flux = np.std(flux)
            normalized = (flux - mean_flux) / (std_flux + 1e-8)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median_flux = np.median(flux)
            q75, q25 = np.percentile(flux, [75, 25])
            iqr = q75 - q25
            normalized = (flux - median_flux) / (iqr + 1e-8)
        
        elif method == 'minmax':
            # Min-max scaling
            min_flux = np.min(flux)
            max_flux = np.max(flux)
            normalized = (flux - min_flux) / (max_flux - min_flux + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def bin_light_curve(
        self, 
        flux: np.ndarray, 
        target_length: int = 2001
    ) -> np.ndarray:
        """Bin light curve to target length.
        
        Args:
            flux: Input flux array
            target_length: Target length
            
        Returns:
            Binned flux array
        """
        if len(flux) == target_length:
            return flux
        
        if len(flux) < target_length:
            # Pad with median value
            median_val = np.median(flux)
            padded = np.full(target_length, median_val)
            padded[:len(flux)] = flux
            return padded
        
        else:
            # Bin down using averaging
            bin_size = len(flux) // target_length
            remainder = len(flux) % target_length
            
            # Reshape and average
            trimmed = flux[:len(flux) - remainder]
            reshaped = trimmed.reshape(-1, bin_size)
            binned = np.mean(reshaped, axis=1)
            
            # Pad if needed
            if len(binned) < target_length:
                padded = np.full(target_length, np.median(binned))
                padded[:len(binned)] = binned
                return padded
            
            return binned[:target_length]
    
    def process_light_curves(
        self, 
        target_ids: List[str], 
        mission: str,
        normalize_method: str = 'robust',
        target_length: int = 2001
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Process multiple light curves.
        
        Args:
            target_ids: List of target identifiers
            mission: Mission name
            normalize_method: Normalization method
            target_length: Target length for binning
            
        Returns:
            Tuple of (processed_light_curves, valid_target_ids)
        """
        processed_curves = []
        valid_ids = []
        
        mission_dir = self.raw_dir / mission.lower()
        
        for target_id in target_ids:
            # Find FITS file
            fits_pattern = f"*{target_id.replace(' ', '_')}*.fits"
            fits_files = list(mission_dir.glob(fits_pattern))
            
            if not fits_files:
                logger.warning(f"No FITS file found for {target_id}")
                continue
            
            # Load light curve
            flux = self.load_light_curve_from_fits(fits_files[0])
            
            if flux is None:
                continue
            
            # Normalize
            flux = self.normalize_light_curve(flux, method=normalize_method)
            
            # Bin to target length
            flux = self.bin_light_curve(flux, target_length=target_length)
            
            processed_curves.append(flux)
            valid_ids.append(target_id)
        
        logger.info(f"Successfully processed {len(processed_curves)}/{len(target_ids)} light curves")
        
        return processed_curves, valid_ids
    
    def create_datasets(
        self, 
        mission: str,
        max_samples: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[LightCurveDataset, LightCurveDataset, LightCurveDataset]:
        """Create train/val/test datasets.
        
        Args:
            mission: Mission name
            max_samples: Maximum number of samples
            test_size: Test set fraction
            val_size: Validation set fraction
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Get labels
        labels_dict = self.catalog_manager.get_labels(mission)
        
        # Get target IDs
        target_ids = list(labels_dict.keys())
        
        if max_samples and len(target_ids) > max_samples:
            np.random.seed(42)
            target_ids = np.random.choice(target_ids, max_samples, replace=False).tolist()
        
        # Process light curves
        light_curves, valid_ids = self.process_light_curves(target_ids, mission)
        
        if not light_curves:
            raise ValueError("No valid light curves processed")
        
        # Get labels for valid IDs
        valid_labels = [labels_dict[tid] for tid in valid_ids]
        
        # Create train/val/test splits
        train_ids, val_ids, test_ids, train_labels, val_labels, test_labels = \
            self.catalog_manager.create_train_test_split(
                valid_ids, labels_dict, test_size=test_size, val_size=val_size
            )
        
        # Split light curves
        id_to_idx = {tid: i for i, tid in enumerate(valid_ids)}
        
        train_curves = [light_curves[id_to_idx[tid]] for tid in train_ids]
        val_curves = [light_curves[id_to_idx[tid]] for tid in val_ids] if val_ids else []
        test_curves = [light_curves[id_to_idx[tid]] for tid in test_ids]
        
        # Create datasets
        train_dataset = LightCurveDataset(
            train_curves, train_labels, target_ids=train_ids, augment=True
        )
        
        val_dataset = LightCurveDataset(
            val_curves, val_labels, target_ids=val_ids, augment=False
        ) if val_curves else None
        
        test_dataset = LightCurveDataset(
            test_curves, test_labels, target_ids=test_ids, augment=False
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Compute class weights for imbalanced data.
        
        Args:
            labels: List of labels
            
        Returns:
            Class weights tensor
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        return torch.FloatTensor(class_weights)
    
    def create_data_loaders(
        self,
        train_dataset: LightCurveDataset,
        val_dataset: Optional[LightCurveDataset],
        test_dataset: LightCurveDataset,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Create PyTorch DataLoaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            Tuple of DataLoaders
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ) if val_dataset else None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader