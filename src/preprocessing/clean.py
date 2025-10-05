"""
Light curve cleaning utilities.
"""

import numpy as np
from typing import Tuple, Optional
import warnings
from scipy import stats

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class LightCurveCleaner:
    """Clean and filter light curve data."""
    
    def __init__(self):
        """Initialize cleaner with config."""
        self.config = config_loader.load('config')
        self.preproc_config = self.config['preprocessing']
    
    def remove_nans(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove NaN and infinite values.
        
        Args:
            time: Time array
            flux: Flux array
            
        Returns:
            Cleaned time and flux arrays
        """
        # Find finite values
        finite_mask = np.isfinite(time) & np.isfinite(flux)
        
        if not np.any(finite_mask):
            logger.warning("No finite values found in light curve")
            return np.array([]), np.array([])
        
        n_removed = len(time) - np.sum(finite_mask)
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} NaN/infinite values")
        
        return time[finite_mask], flux[finite_mask]
    
    def remove_outliers(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        sigma: float = 5.0,
        method: str = 'sigma_clip'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from light curve.
        
        Args:
            time: Time array
            flux: Flux array
            sigma: Sigma threshold for outlier detection
            method: Outlier detection method ('sigma_clip', 'iqr')
            
        Returns:
            Cleaned time and flux arrays
        """
        if len(flux) == 0:
            return time, flux
        
        if method == 'sigma_clip':
            # Sigma clipping
            median_flux = np.median(flux)
            mad = np.median(np.abs(flux - median_flux))
            
            # Convert MAD to standard deviation estimate
            std_estimate = 1.4826 * mad
            
            # Create mask for values within sigma threshold
            z_scores = np.abs(flux - median_flux) / (std_estimate + 1e-8)
            outlier_mask = z_scores < sigma
        
        elif method == 'iqr':
            # Interquartile range method
            q1, q3 = np.percentile(flux, [25, 75])
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (flux >= lower_bound) & (flux <= upper_bound)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        n_removed = len(flux) - np.sum(outlier_mask)
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} outliers using {method} method")
        
        return time[outlier_mask], flux[outlier_mask]
    
    def remove_short_segments(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        min_length: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove light curves that are too short.
        
        Args:
            time: Time array
            flux: Flux array
            min_length: Minimum required length
            
        Returns:
            Time and flux arrays (empty if too short)
        """
        if len(flux) < min_length:
            logger.warning(f"Light curve too short: {len(flux)} < {min_length}")
            return np.array([]), np.array([])
        
        return time, flux
    
    def remove_gaps(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        max_gap: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove large gaps in time series.
        
        Args:
            time: Time array
            flux: Flux array
            max_gap: Maximum allowed gap in time units
            
        Returns:
            Time and flux arrays with large gaps removed
        """
        if len(time) < 2:
            return time, flux
        
        # Calculate time differences
        time_diffs = np.diff(time)
        
        # Find large gaps
        gap_mask = time_diffs > max_gap
        
        if not np.any(gap_mask):
            return time, flux
        
        # Find the longest continuous segment
        gap_indices = np.where(gap_mask)[0]
        
        # Add boundaries
        boundaries = np.concatenate([[0], gap_indices + 1, [len(time)]])
        
        # Find longest segment
        segment_lengths = np.diff(boundaries)
        longest_segment_idx = np.argmax(segment_lengths)
        
        start_idx = boundaries[longest_segment_idx]
        end_idx = boundaries[longest_segment_idx + 1]
        
        logger.debug(f"Kept longest segment: {end_idx - start_idx} points")
        
        return time[start_idx:end_idx], flux[start_idx:end_idx]
    
    def clean_light_curve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        remove_outliers: bool = True,
        outlier_sigma: float = 5.0,
        min_length: int = 100,
        max_gap: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply full cleaning pipeline to light curve.
        
        Args:
            time: Time array
            flux: Flux array
            remove_outliers: Whether to remove outliers
            outlier_sigma: Sigma threshold for outliers
            min_length: Minimum length requirement
            max_gap: Maximum time gap allowed
            
        Returns:
            Cleaned time and flux arrays
        """
        original_length = len(flux)
        
        # Step 1: Remove NaNs
        time, flux = self.remove_nans(time, flux)
        
        if len(flux) == 0:
            return time, flux
        
        # Step 2: Remove large gaps
        time, flux = self.remove_gaps(time, flux, max_gap=max_gap)
        
        # Step 3: Check minimum length
        time, flux = self.remove_short_segments(time, flux, min_length=min_length)
        
        if len(flux) == 0:
            return time, flux
        
        # Step 4: Remove outliers
        if remove_outliers:
            time, flux = self.remove_outliers(time, flux, sigma=outlier_sigma)
        
        final_length = len(flux)
        removed_fraction = (original_length - final_length) / original_length
        
        logger.debug(f"Cleaning: {original_length} -> {final_length} points ({removed_fraction:.1%} removed)")
        
        return time, flux
    
    def validate_light_curve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> bool:
        """Validate light curve quality.
        
        Args:
            time: Time array
            flux: Flux array
            
        Returns:
            True if light curve passes quality checks
        """
        if len(flux) == 0:
            return False
        
        # Check for minimum length
        if len(flux) < 100:
            logger.debug("Light curve too short")
            return False
        
        # Check for sufficient time coverage
        if len(time) > 1:
            time_span = np.max(time) - np.min(time)
            if time_span < 1.0:  # Less than 1 day
                logger.debug("Time span too short")
                return False
        
        # Check for sufficient flux variation
        flux_std = np.std(flux)
        flux_median = np.median(flux)
        
        if flux_std / (abs(flux_median) + 1e-8) < 1e-6:
            logger.debug("Insufficient flux variation")
            return False
        
        # Check flux range
        if np.max(flux) == np.min(flux):
            logger.debug("No flux variation")
            return False
        
        return True