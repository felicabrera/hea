"""
Light curve detrending utilities.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import signal
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from scipy.stats import binned_statistic

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class LightCurveDetrender:
    """Detrend light curves to remove instrumental and stellar trends."""
    
    def __init__(self):
        """Initialize detrender with config."""
        self.config = config_loader.load('config')
        self.preproc_config = self.config['preprocessing']
    
    def savgol_detrend(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detrend using Savitzky-Golay filter.
        
        Args:
            time: Time array
            flux: Flux array
            window_length: Window length for filter
            polyorder: Polynomial order
            
        Returns:
            Detrended time and flux
        """
        if len(flux) < 10:
            return time, flux
        
        # Auto-determine window length if not provided
        if window_length is None:
            window_length = min(101, len(flux) // 4)
            if window_length % 2 == 0:  # Must be odd
                window_length += 1
            window_length = max(polyorder + 1, window_length)
        
        try:
            # Apply Savitzky-Golay filter to get trend
            trend = signal.savgol_filter(flux, window_length, polyorder)
            
            # Remove trend
            detrended_flux = flux - trend + np.median(flux)
            
            logger.debug(f"Applied Savitzky-Golay detrending (window={window_length}, poly={polyorder})")
            
            return time, detrended_flux
            
        except Exception as e:
            logger.warning(f"Savitzky-Golay detrending failed: {e}")
            return time, flux
    
    def spline_detrend(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        smoothing_factor: Optional[float] = None,
        spline_degree: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detrend using spline fitting.
        
        Args:
            time: Time array
            flux: Flux array
            smoothing_factor: Smoothing parameter for spline
            spline_degree: Degree of spline
            
        Returns:
            Detrended time and flux
        """
        if len(flux) < 10:
            return time, flux
        
        try:
            # Auto-determine smoothing factor
            if smoothing_factor is None:
                smoothing_factor = len(flux) * np.var(flux) * 0.1
            
            # Fit spline
            spline = UnivariateSpline(time, flux, s=smoothing_factor, k=spline_degree)
            trend = spline(time)
            
            # Remove trend
            detrended_flux = flux - trend + np.median(flux)
            
            logger.debug(f"Applied spline detrending (s={smoothing_factor:.2e}, k={spline_degree})")
            
            return time, detrended_flux
            
        except Exception as e:
            logger.warning(f"Spline detrending failed: {e}")
            return time, flux
    
    def median_detrend(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        window_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detrend using median filter.
        
        Args:
            time: Time array
            flux: Flux array
            window_size: Size of median filter window
            
        Returns:
            Detrended time and flux
        """
        if len(flux) < 10:
            return time, flux
        
        # Auto-determine window size
        if window_size is None:
            window_size = min(51, len(flux) // 10)
            if window_size % 2 == 0:  # Should be odd
                window_size += 1
            window_size = max(3, window_size)
        
        try:
            # Apply median filter to get trend
            trend = median_filter(flux, size=window_size, mode='reflect')
            
            # Remove trend
            detrended_flux = flux - trend + np.median(flux)
            
            logger.debug(f"Applied median filter detrending (window={window_size})")
            
            return time, detrended_flux
            
        except Exception as e:
            logger.warning(f"Median filter detrending failed: {e}")
            return time, flux
    
    def polynomial_detrend(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        degree: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detrend using polynomial fitting.
        
        Args:
            time: Time array
            flux: Flux array
            degree: Polynomial degree
            
        Returns:
            Detrended time and flux
        """
        if len(flux) <= degree + 1:
            return time, flux
        
        try:
            # Fit polynomial
            coeffs = np.polyfit(time, flux, degree)
            trend = np.polyval(coeffs, time)
            
            # Remove trend
            detrended_flux = flux - trend + np.median(flux)
            
            logger.debug(f"Applied polynomial detrending (degree={degree})")
            
            return time, detrended_flux
            
        except Exception as e:
            logger.warning(f"Polynomial detrending failed: {e}")
            return time, flux
    
    def biweight_detrend(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        window_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detrend using biweight location in sliding window.
        
        Args:
            time: Time array
            flux: Flux array
            window_size: Size of sliding window
            
        Returns:
            Detrended time and flux
        """
        if len(flux) < 10:
            return time, flux
        
        # Auto-determine window size
        if window_size is None:
            window_size = min(101, len(flux) // 5)
            window_size = max(21, window_size)
        
        try:
            from astropy.stats import biweight_location
            
            # Calculate biweight trend using sliding window
            half_window = window_size // 2
            trend = np.zeros_like(flux)
            
            for i in range(len(flux)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(flux), i + half_window + 1)
                
                window_flux = flux[start_idx:end_idx]
                trend[i] = biweight_location(window_flux)
            
            # Remove trend
            detrended_flux = flux - trend + np.median(flux)
            
            logger.debug(f"Applied biweight detrending (window={window_size})")
            
            return time, detrended_flux
            
        except ImportError:
            logger.warning("Astropy not available, falling back to median detrending")
            return self.median_detrend(time, flux, window_size)
        except Exception as e:
            logger.warning(f"Biweight detrending failed: {e}")
            return time, flux
    
    def detrend_light_curve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        method: str = 'savgol',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply detrending to light curve.
        
        Args:
            time: Time array
            flux: Flux array
            method: Detrending method
            **kwargs: Additional parameters for detrending method
            
        Returns:
            Detrended time and flux
        """
        if len(flux) == 0:
            return time, flux
        
        method = method.lower()
        
        if method == 'savgol':
            return self.savgol_detrend(time, flux, **kwargs)
        elif method == 'spline':
            return self.spline_detrend(time, flux, **kwargs)
        elif method == 'median':
            return self.median_detrend(time, flux, **kwargs)
        elif method == 'polynomial':
            return self.polynomial_detrend(time, flux, **kwargs)
        elif method == 'biweight':
            return self.biweight_detrend(time, flux, **kwargs)
        elif method == 'none':
            logger.debug("No detrending applied")
            return time, flux
        else:
            logger.warning(f"Unknown detrending method: {method}, applying savgol")
            return self.savgol_detrend(time, flux, **kwargs)
    
    def auto_detrend(
        self, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Automatically select best detrending method.
        
        Args:
            time: Time array
            flux: Flux array
            
        Returns:
            Tuple of (detrended_time, detrended_flux, method_used)
        """
        if len(flux) < 20:
            return time, flux, 'none'
        
        methods = ['savgol', 'median', 'polynomial']
        best_method = 'savgol'
        best_score = float('inf')
        best_result = (time, flux)
        
        # Calculate original flux statistics
        original_std = np.std(flux)
        
        for method in methods:
            try:
                _, detrended_flux = self.detrend_light_curve(time, flux, method=method)
                
                # Score based on how much we reduced variability
                # while preserving potential transit signals
                detrended_std = np.std(detrended_flux)
                
                # Penalize over-smoothing (too much std reduction)
                # and under-smoothing (too little std reduction)
                std_ratio = detrended_std / original_std
                
                if 0.3 <= std_ratio <= 0.9:  # Good range
                    score = abs(std_ratio - 0.6)  # Prefer ~60% of original std
                else:
                    score = 1.0  # Poor score
                
                if score < best_score:
                    best_score = score
                    best_method = method
                    best_result = (time, detrended_flux)
                    
            except Exception:
                continue
        
        logger.debug(f"Auto-selected detrending method: {best_method}")
        
        return best_result[0], best_result[1], best_method