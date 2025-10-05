"""
Complete preprocessing pipeline for light curve data.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import pickle
import json

from .clean import LightCurveCleaner
from .detrend import LightCurveDetrender
from .normalize import LightCurveNormalizer
from .folding import PhaseFolder
from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for exoplanet light curves."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing pipeline.
        
        Args:
            config_override: Optional config overrides
        """
        self.config = config_loader.load('config')
        
        # Apply config overrides
        if config_override:
            self.config.update(config_override)
        
        self.preproc_config = self.config['preprocessing']
        
        # Initialize components
        self.cleaner = LightCurveCleaner()
        self.detrender = LightCurveDetrender()
        self.normalizer = LightCurveNormalizer()
        self.folder = PhaseFolder()
        
        # Store processing parameters
        self.processing_params = {}
    
    def process_single_light_curve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        period: Optional[float] = None,
        epoch: Optional[float] = None,
        target_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single light curve through the full pipeline.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period (required for phase folding)
            epoch: Transit epoch
            target_id: Target identifier for logging
            
        Returns:
            Tuple of (processed_flux, processing_metadata)
        """
        if target_id is None:
            target_id = "unknown"
        
        metadata = {
            'target_id': target_id,
            'original_length': len(flux),
            'processing_steps': [],
            'success': False
        }
        
        try:
            # Step 1: Cleaning
            logger.debug(f"[{target_id}] Starting cleaning...")
            time_clean, flux_clean = self.cleaner.clean_light_curve(
                time, flux,
                remove_outliers=self.preproc_config.get('remove_outliers', True),
                outlier_sigma=self.preproc_config.get('outlier_sigma', 5.0)
            )
            
            if len(flux_clean) == 0:
                logger.warning(f"[{target_id}] Light curve empty after cleaning")
                return self._create_empty_result(metadata)
            
            metadata['processing_steps'].append('cleaning')
            metadata['cleaned_length'] = len(flux_clean)
            
            # Validate cleaned curve
            if not self.cleaner.validate_light_curve(time_clean, flux_clean):
                logger.warning(f"[{target_id}] Light curve failed validation")
                return self._create_empty_result(metadata)
            
            # Step 2: Detrending
            logger.debug(f"[{target_id}] Starting detrending...")
            detrend_method = self.preproc_config.get('detrend_method', 'savgol')
            
            if detrend_method == 'auto':
                time_detrend, flux_detrend, used_method = self.detrender.auto_detrend(
                    time_clean, flux_clean
                )
                metadata['detrend_method'] = used_method
            else:
                time_detrend, flux_detrend = self.detrender.detrend_light_curve(
                    time_clean, flux_clean, method=detrend_method
                )
                metadata['detrend_method'] = detrend_method
            
            metadata['processing_steps'].append('detrending')
            
            # Step 3: Normalization
            logger.debug(f"[{target_id}] Starting normalization...")
            normalize_method = self.preproc_config.get('normalize_method', 'robust')
            
            if normalize_method == 'auto':
                flux_norm, norm_params, used_method = self.normalizer.auto_normalize(
                    flux_detrend
                )
                metadata['normalize_method'] = used_method
            else:
                flux_norm, norm_params = self.normalizer.normalize_light_curve(
                    flux_detrend, method=normalize_method
                )
                metadata['normalize_method'] = normalize_method
            
            metadata['normalization_params'] = norm_params
            metadata['processing_steps'].append('normalization')
            
            # Step 4: Phase folding (if period provided)
            if period is not None and self.preproc_config.get('phase_fold', True):
                logger.debug(f"[{target_id}] Starting phase folding...")
                
                target_length = self.preproc_config.get('time_series_length', 2001)
                phase, flux_folded = self.folder.create_phase_curve(
                    time_detrend, flux_norm, 
                    period=period, 
                    epoch=epoch,
                    n_bins=target_length
                )
                
                processed_flux = flux_folded
                metadata['processing_steps'].append('phase_folding')
                metadata['period'] = period
                metadata['epoch'] = epoch
                metadata['phase_folded'] = True
                
            else:
                # Step 4b: Binning without phase folding
                logger.debug(f"[{target_id}] Binning without phase folding...")
                
                target_length = self.preproc_config.get('time_series_length', 2001)
                processed_flux = self._bin_time_series(flux_norm, target_length)
                
                metadata['processing_steps'].append('binning')
                metadata['phase_folded'] = False
            
            metadata['final_length'] = len(processed_flux)
            metadata['success'] = True
            
            logger.debug(f"[{target_id}] Processing completed successfully")
            
            return processed_flux, metadata
            
        except Exception as e:
            logger.error(f"[{target_id}] Processing failed: {e}")
            metadata['error'] = str(e)
            return self._create_empty_result(metadata)
    
    def _create_empty_result(self, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create empty result for failed processing.
        
        Args:
            metadata: Metadata dictionary to update
            
        Returns:
            Empty result tuple
        """
        target_length = self.preproc_config.get('time_series_length', 2001)
        empty_flux = np.full(target_length, 1.0)  # Flat light curve
        metadata['final_length'] = target_length
        metadata['is_empty'] = True
        return empty_flux, metadata
    
    def _bin_time_series(self, flux: np.ndarray, target_length: int) -> np.ndarray:
        """Bin time series to target length.
        
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
    
    def process_batch(
        self, 
        light_curves: List[Tuple[np.ndarray, np.ndarray]], 
        periods: Optional[List[float]] = None,
        epochs: Optional[List[float]] = None,
        target_ids: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Process a batch of light curves.
        
        Args:
            light_curves: List of (time, flux) tuples
            periods: Optional list of periods
            epochs: Optional list of epochs
            target_ids: Optional list of target IDs
            
        Returns:
            Tuple of (processed_fluxes, metadata_list)
        """
        n_curves = len(light_curves)
        
        # Set defaults
        if periods is None:
            periods = [None] * n_curves
        if epochs is None:
            epochs = [None] * n_curves
        if target_ids is None:
            target_ids = [f"target_{i}" for i in range(n_curves)]
        
        processed_fluxes = []
        metadata_list = []
        
        logger.info(f"Processing batch of {n_curves} light curves...")
        
        for i, ((time, flux), period, epoch, target_id) in enumerate(
            zip(light_curves, periods, epochs, target_ids)
        ):
            if i % 100 == 0:
                logger.info(f"Processing curve {i+1}/{n_curves}")
            
            processed_flux, metadata = self.process_single_light_curve(
                time, flux, period, epoch, target_id
            )
            
            processed_fluxes.append(processed_flux)
            metadata_list.append(metadata)
        
        # Summary statistics
        successful = sum(1 for meta in metadata_list if meta['success'])
        logger.info(f"Batch processing completed: {successful}/{n_curves} successful")
        
        return processed_fluxes, metadata_list
    
    def save_processing_params(self, filepath: Path) -> None:
        """Save processing parameters to file.
        
        Args:
            filepath: Path to save parameters
        """
        params = {
            'config': self.config,
            'processing_params': self.processing_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        
        logger.info(f"Saved processing parameters to {filepath}")
    
    def load_processing_params(self, filepath: Path) -> None:
        """Load processing parameters from file.
        
        Args:
            filepath: Path to load parameters from
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.config.update(params['config'])
        self.processing_params = params['processing_params']
        
        logger.info(f"Loaded processing parameters from {filepath}")
    
    def get_processing_summary(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate processing summary statistics.
        
        Args:
            metadata_list: List of processing metadata
            
        Returns:
            Summary statistics dictionary
        """
        total_curves = len(metadata_list)
        successful = sum(1 for meta in metadata_list if meta['success'])
        failed = total_curves - successful
        
        # Processing step statistics
        step_counts = {}
        for meta in metadata_list:
            for step in meta.get('processing_steps', []):
                step_counts[step] = step_counts.get(step, 0) + 1
        
        # Method usage statistics
        detrend_methods = {}
        normalize_methods = {}
        
        for meta in metadata_list:
            if 'detrend_method' in meta:
                method = meta['detrend_method']
                detrend_methods[method] = detrend_methods.get(method, 0) + 1
            
            if 'normalize_method' in meta:
                method = meta['normalize_method']
                normalize_methods[method] = normalize_methods.get(method, 0) + 1
        
        # Length statistics
        original_lengths = [meta.get('original_length', 0) for meta in metadata_list]
        final_lengths = [meta.get('final_length', 0) for meta in metadata_list]
        
        summary = {
            'total_curves': total_curves,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_curves if total_curves > 0 else 0,
            'processing_steps': step_counts,
            'detrend_methods': detrend_methods,
            'normalize_methods': normalize_methods,
            'length_stats': {
                'original_mean': np.mean(original_lengths) if original_lengths else 0,
                'original_std': np.std(original_lengths) if original_lengths else 0,
                'final_mean': np.mean(final_lengths) if final_lengths else 0,
                'final_std': np.std(final_lengths) if final_lengths else 0
            }
        }
        
        return summary
    
    def validate_pipeline(self) -> bool:
        """Validate pipeline configuration and components.
        
        Returns:
            True if pipeline is valid
        """
        try:
            # Test with synthetic data
            time = np.linspace(0, 10, 1000)
            flux = np.ones_like(time) + 0.01 * np.random.randn(len(time))
            
            # Add a simple transit
            transit_mask = (time > 4.9) & (time < 5.1)
            flux[transit_mask] *= 0.99  # 1% depth transit
            
            processed_flux, metadata = self.process_single_light_curve(
                time, flux, period=2.0, target_id="validation_test"
            )
            
            # Check result
            if not metadata['success']:
                logger.error("Pipeline validation failed: processing unsuccessful")
                return False
            
            if len(processed_flux) != self.preproc_config.get('time_series_length', 2001):
                logger.error("Pipeline validation failed: incorrect output length")
                return False
            
            logger.info("Pipeline validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return False