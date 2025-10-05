"""
Light curve normalization utilities.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class LightCurveNormalizer:
    """Normalize light curve flux values."""
    
    def __init__(self):
        """Initialize normalizer with config."""
        self.config = config_loader.load('config')
        self.preproc_config = self.config['preprocessing']
    
    def standard_normalize(
        self, 
        flux: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Standard normalization (z-score).
        
        Args:
            flux: Flux array
            
        Returns:
            Tuple of (normalized_flux, normalization_params)
        """
        if len(flux) == 0:
            return flux, {}
        
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        
        if std_flux == 0:
            logger.warning("Zero standard deviation, returning original flux")
            return flux, {'mean': mean_flux, 'std': 1.0, 'method': 'standard'}
        
        normalized = (flux - mean_flux) / std_flux
        
        params = {
            'mean': mean_flux,
            'std': std_flux,
            'method': 'standard'
        }
        
        logger.debug(f"Applied standard normalization (mean={mean_flux:.4f}, std={std_flux:.4f})")
        
        return normalized, params
    
    def robust_normalize(
        self, 
        flux: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Robust normalization using median and MAD.
        
        Args:
            flux: Flux array
            
        Returns:
            Tuple of (normalized_flux, normalization_params)
        """
        if len(flux) == 0:
            return flux, {}
        
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        
        # Convert MAD to standard deviation equivalent
        mad_to_std = 1.4826
        robust_std = mad * mad_to_std
        
        if robust_std == 0:
            logger.warning("Zero robust standard deviation, returning original flux")
            return flux, {'median': median_flux, 'mad': mad, 'method': 'robust'}
        
        normalized = (flux - median_flux) / robust_std
        
        params = {
            'median': median_flux,
            'mad': mad,
            'robust_std': robust_std,
            'method': 'robust'
        }
        
        logger.debug(f"Applied robust normalization (median={median_flux:.4f}, mad={mad:.4f})")
        
        return normalized, params
    
    def minmax_normalize(
        self, 
        flux: np.ndarray,
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[np.ndarray, dict]:
        """Min-max normalization.
        
        Args:
            flux: Flux array
            feature_range: Target range for normalization
            
        Returns:
            Tuple of (normalized_flux, normalization_params)
        """
        if len(flux) == 0:
            return flux, {}
        
        min_flux = np.min(flux)
        max_flux = np.max(flux)
        
        flux_range = max_flux - min_flux
        
        if flux_range == 0:
            logger.warning("Zero flux range, returning constant array")
            mid_point = (feature_range[0] + feature_range[1]) / 2
            return np.full_like(flux, mid_point), {
                'min': min_flux, 'max': max_flux, 'range': 0, 'method': 'minmax'
            }
        
        # Scale to [0, 1] first
        normalized = (flux - min_flux) / flux_range
        
        # Scale to desired range
        target_min, target_max = feature_range
        target_range = target_max - target_min
        normalized = normalized * target_range + target_min
        
        params = {
            'min': min_flux,
            'max': max_flux,
            'range': flux_range,
            'feature_range': feature_range,
            'method': 'minmax'
        }
        
        logger.debug(f"Applied min-max normalization (range=[{min_flux:.4f}, {max_flux:.4f}])")
        
        return normalized, params
    
    def quantile_normalize(
        self, 
        flux: np.ndarray,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95
    ) -> Tuple[np.ndarray, dict]:
        """Quantile-based normalization.
        
        Args:
            flux: Flux array
            lower_quantile: Lower quantile for normalization
            upper_quantile: Upper quantile for normalization
            
        Returns:
            Tuple of (normalized_flux, normalization_params)
        """
        if len(flux) == 0:
            return flux, {}
        
        q_low = np.quantile(flux, lower_quantile)
        q_high = np.quantile(flux, upper_quantile)
        
        q_range = q_high - q_low
        
        if q_range == 0:
            logger.warning("Zero quantile range, falling back to standard normalization")
            return self.standard_normalize(flux)
        
        # Clip to quantile range and normalize
        clipped_flux = np.clip(flux, q_low, q_high)
        normalized = (clipped_flux - q_low) / q_range
        
        params = {
            'q_low': q_low,
            'q_high': q_high,
            'q_range': q_range,
            'lower_quantile': lower_quantile,
            'upper_quantile': upper_quantile,
            'method': 'quantile'
        }
        
        logger.debug(f"Applied quantile normalization (q{lower_quantile}={q_low:.4f}, q{upper_quantile}={q_high:.4f})")
        
        return normalized, params
    
    def unit_normalize(
        self, 
        flux: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Unit vector normalization (L2 norm).
        
        Args:
            flux: Flux array
            
        Returns:
            Tuple of (normalized_flux, normalization_params)
        """
        if len(flux) == 0:
            return flux, {}
        
        l2_norm = np.linalg.norm(flux)
        
        if l2_norm == 0:
            logger.warning("Zero L2 norm, returning original flux")
            return flux, {'l2_norm': l2_norm, 'method': 'unit'}
        
        normalized = flux / l2_norm
        
        params = {
            'l2_norm': l2_norm,
            'method': 'unit'
        }
        
        logger.debug(f"Applied unit normalization (L2 norm={l2_norm:.4f})")
        
        return normalized, params
    
    def normalize_light_curve(
        self, 
        flux: np.ndarray,
        method: str = 'robust',
        **kwargs
    ) -> Tuple[np.ndarray, dict]:
        """Apply normalization to light curve.
        
        Args:
            flux: Flux array
            method: Normalization method
            **kwargs: Additional parameters for normalization method
            
        Returns:
            Tuple of (normalized_flux, normalization_params)
        """
        if len(flux) == 0:
            return flux, {}
        
        method = method.lower()
        
        if method == 'standard':
            return self.standard_normalize(flux)
        elif method == 'robust':
            return self.robust_normalize(flux)
        elif method == 'minmax':
            return self.minmax_normalize(flux, **kwargs)
        elif method == 'quantile':
            return self.quantile_normalize(flux, **kwargs)
        elif method == 'unit':
            return self.unit_normalize(flux)
        elif method == 'none':
            logger.debug("No normalization applied")
            return flux, {'method': 'none'}
        else:
            logger.warning(f"Unknown normalization method: {method}, applying robust")
            return self.robust_normalize(flux)
    
    def denormalize(
        self, 
        normalized_flux: np.ndarray, 
        params: dict
    ) -> np.ndarray:
        """Reverse normalization using stored parameters.
        
        Args:
            normalized_flux: Normalized flux array
            params: Normalization parameters from normalize_light_curve
            
        Returns:
            Original scale flux array
        """
        if len(normalized_flux) == 0 or not params:
            return normalized_flux
        
        method = params.get('method', 'none')
        
        if method == 'standard':
            return normalized_flux * params['std'] + params['mean']
        
        elif method == 'robust':
            return normalized_flux * params['robust_std'] + params['median']
        
        elif method == 'minmax':
            # Reverse the scaling
            target_min, target_max = params['feature_range']
            target_range = target_max - target_min
            
            # Scale back from target range to [0, 1]
            unit_scale = (normalized_flux - target_min) / target_range
            
            # Scale back to original range
            return unit_scale * params['range'] + params['min']
        
        elif method == 'quantile':
            return normalized_flux * params['q_range'] + params['q_low']
        
        elif method == 'unit':
            return normalized_flux * params['l2_norm']
        
        elif method == 'none':
            return normalized_flux
        
        else:
            logger.warning(f"Unknown method for denormalization: {method}")
            return normalized_flux
    
    def auto_normalize(
        self, 
        flux: np.ndarray
    ) -> Tuple[np.ndarray, dict, str]:
        """Automatically select best normalization method.
        
        Args:
            flux: Flux array
            
        Returns:
            Tuple of (normalized_flux, params, method_used)
        """
        if len(flux) < 10:
            return flux, {'method': 'none'}, 'none'
        
        # Check for outliers using IQR
        q1, q3 = np.percentile(flux, [25, 75])
        iqr = q3 - q1
        
        if iqr == 0:
            return flux, {'method': 'none'}, 'none'
        
        # Count outliers
        outlier_bounds = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
        n_outliers = np.sum((flux < outlier_bounds[0]) | (flux > outlier_bounds[1]))
        outlier_fraction = n_outliers / len(flux)
        
        # Select method based on outlier fraction
        if outlier_fraction > 0.1:  # Many outliers
            method = 'robust'
        elif outlier_fraction < 0.02:  # Few outliers
            method = 'standard'
        else:  # Moderate outliers
            method = 'quantile'
        
        normalized, params = self.normalize_light_curve(flux, method=method)
        
        logger.debug(f"Auto-selected normalization method: {method} (outlier_fraction={outlier_fraction:.3f})")
        
        return normalized, params, method