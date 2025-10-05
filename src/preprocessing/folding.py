"""
Phase folding utilities for light curves.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.optimize import minimize_scalar
from scipy.stats import binned_statistic

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class PhaseFolder:
    """Phase fold light curves based on orbital period."""
    
    def __init__(self):
        """Initialize phase folder with config."""
        self.config = config_loader.load('config')
        self.preproc_config = self.config['preprocessing']
    
    def fold_light_curve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        period: float,
        epoch: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fold light curve to orbital phase.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            epoch: Reference epoch (time of transit center)
            
        Returns:
            Tuple of (phase, folded_flux, folded_time)
        """
        if len(time) == 0 or period <= 0:
            return np.array([]), np.array([]), np.array([])
        
        # Use first time point as epoch if not provided
        if epoch is None:
            epoch = time[0]
        
        # Calculate phase
        phase = ((time - epoch) / period) % 1.0
        
        # Center phase around 0 (transit at phase 0)
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        
        # Sort by phase
        sort_indices = np.argsort(phase)
        phase_sorted = phase[sort_indices]
        flux_sorted = flux[sort_indices]
        time_sorted = time[sort_indices]
        
        logger.debug(f"Folded light curve with period={period:.4f} days, epoch={epoch:.4f}")
        
        return phase_sorted, flux_sorted, time_sorted
    
    def bin_folded_curve(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray,
        n_bins: int = 1000,
        bin_method: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin phase-folded light curve.
        
        Args:
            phase: Phase array
            flux: Flux array
            n_bins: Number of phase bins
            bin_method: Binning method ('mean', 'median')
            
        Returns:
            Tuple of (bin_centers, binned_flux, bin_errors)
        """
        if len(phase) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Define phase bins
        phase_min, phase_max = -0.5, 0.5
        bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Bin the data
        if bin_method == 'mean':
            binned_flux, _, _ = binned_statistic(
                phase, flux, statistic='mean', bins=bin_edges
            )
            bin_errors, _, _ = binned_statistic(
                phase, flux, statistic='std', bins=bin_edges
            )
        elif bin_method == 'median':
            binned_flux, _, _ = binned_statistic(
                phase, flux, statistic='median', bins=bin_edges
            )
            # Use MAD for errors
            def mad_func(x):
                if len(x) == 0:
                    return np.nan
                return np.median(np.abs(x - np.median(x)))
            
            bin_errors, _, _ = binned_statistic(
                phase, flux, statistic=mad_func, bins=bin_edges
            )
        else:
            raise ValueError(f"Unknown binning method: {bin_method}")
        
        # Remove empty bins
        valid_mask = np.isfinite(binned_flux)
        bin_centers = bin_centers[valid_mask]
        binned_flux = binned_flux[valid_mask]
        bin_errors = bin_errors[valid_mask]
        
        # Fill NaN errors with median error
        if np.any(np.isnan(bin_errors)):
            median_error = np.nanmedian(bin_errors)
            bin_errors = np.where(np.isnan(bin_errors), median_error, bin_errors)
        
        logger.debug(f"Binned folded curve to {len(bin_centers)} points using {bin_method}")
        
        return bin_centers, binned_flux, bin_errors
    
    def find_transit_epoch(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        period: float,
        epoch_guess: Optional[float] = None
    ) -> float:
        """Find optimal transit epoch by minimizing folded light curve.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            epoch_guess: Initial guess for epoch
            
        Returns:
            Optimized epoch
        """
        if len(time) == 0 or period <= 0:
            return time[0] if len(time) > 0 else 0.0
        
        # Use middle of time series as initial guess
        if epoch_guess is None:
            epoch_guess = (time[0] + time[-1]) / 2
        
        def objective(epoch):
            """Objective function: minimize flux at phase 0."""
            phase = ((time - epoch) / period) % 1.0
            phase = np.where(phase > 0.5, phase - 1.0, phase)
            
            # Find points near phase 0 (transit)
            transit_mask = np.abs(phase) < 0.1
            
            if not np.any(transit_mask):
                return 1.0  # No transit points found
            
            # Return negative of minimum flux (we want to minimize)
            return -np.min(flux[transit_mask])
        
        try:
            # Optimize epoch within one period around the guess
            bounds = (epoch_guess - period/2, epoch_guess + period/2)
            result = minimize_scalar(objective, bounds=bounds, method='bounded')
            
            optimal_epoch = result.x
            
            logger.debug(f"Optimized epoch from {epoch_guess:.4f} to {optimal_epoch:.4f}")
            
            return optimal_epoch
            
        except Exception as e:
            logger.warning(f"Epoch optimization failed: {e}, using guess")
            return epoch_guess
    
    def create_phase_curve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        period: float,
        epoch: Optional[float] = None,
        n_bins: int = 2001,
        optimize_epoch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a standardized phase curve for ML input.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            epoch: Reference epoch
            n_bins: Number of phase bins for output
            optimize_epoch: Whether to optimize epoch
            
        Returns:
            Tuple of (phase_array, binned_flux_array)
        """
        if len(time) == 0:
            # Return empty arrays with correct length
            phase_array = np.linspace(-0.5, 0.5, n_bins)
            flux_array = np.full(n_bins, 1.0)  # Flat light curve
            return phase_array, flux_array
        
        # Optimize epoch if requested
        if optimize_epoch:
            epoch = self.find_transit_epoch(time, flux, period, epoch)
        elif epoch is None:
            epoch = time[0]
        
        # Fold light curve
        phase, folded_flux, _ = self.fold_light_curve(time, flux, period, epoch)
        
        # Bin to standard resolution
        bin_centers, binned_flux, _ = self.bin_folded_curve(
            phase, folded_flux, n_bins=n_bins
        )
        
        # Create regular phase array
        phase_array = np.linspace(-0.5, 0.5, n_bins)
        
        # Interpolate to regular grid if needed
        if len(bin_centers) != n_bins:
            flux_array = np.interp(phase_array, bin_centers, binned_flux)
        else:
            flux_array = binned_flux
        
        # Fill any remaining NaN values with median
        if np.any(np.isnan(flux_array)):
            median_flux = np.nanmedian(flux_array)
            flux_array = np.where(np.isnan(flux_array), median_flux, flux_array)
        
        logger.debug(f"Created phase curve with {n_bins} points")
        
        return phase_array, flux_array
    
    def detect_secondary_eclipse(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray,
        primary_phase: float = 0.0
    ) -> Tuple[bool, float]:
        """Detect secondary eclipse in phase-folded data.
        
        Args:
            phase: Phase array
            flux: Flux array
            primary_phase: Phase of primary transit
            
        Returns:
            Tuple of (eclipse_detected, eclipse_phase)
        """
        if len(phase) == 0:
            return False, 0.5
        
        # Expected secondary eclipse phase (opposite side of orbit)
        expected_secondary = primary_phase + 0.5
        if expected_secondary > 0.5:
            expected_secondary -= 1.0
        
        # Search for secondary eclipse in a window around expected phase
        search_window = 0.1
        search_mask = np.abs(phase - expected_secondary) < search_window
        
        if not np.any(search_mask):
            return False, expected_secondary
        
        # Find minimum flux in search window
        search_phase = phase[search_mask]
        search_flux = flux[search_mask]
        
        min_idx = np.argmin(search_flux)
        secondary_phase = search_phase[min_idx]
        secondary_depth = search_flux[min_idx]
        
        # Check if it's a significant dip
        median_flux = np.median(flux)
        flux_std = np.std(flux)
        
        # Require at least 2-sigma detection
        detection_threshold = median_flux - 2 * flux_std
        
        eclipse_detected = secondary_depth < detection_threshold
        
        if eclipse_detected:
            logger.debug(f"Secondary eclipse detected at phase {secondary_phase:.3f}")
        
        return eclipse_detected, secondary_phase
    
    def fold_multiple_periods(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        periods: List[float],
        epochs: Optional[List[float]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Fold light curve with multiple periods.
        
        Args:
            time: Time array
            flux: Flux array
            periods: List of periods to try
            epochs: Optional list of epochs for each period
            
        Returns:
            List of (phase, flux) tuples for each period
        """
        if epochs is None:
            epochs = [None] * len(periods)
        
        results = []
        
        for period, epoch in zip(periods, epochs):
            try:
                phase_curve = self.create_phase_curve(
                    time, flux, period, epoch, optimize_epoch=(epoch is None)
                )
                results.append(phase_curve)
            except Exception as e:
                logger.warning(f"Failed to fold with period {period}: {e}")
                # Return empty result
                n_bins = self.preproc_config.get('time_series_length', 2001)
                empty_phase = np.linspace(-0.5, 0.5, n_bins)
                empty_flux = np.full(n_bins, 1.0)
                results.append((empty_phase, empty_flux))
        
        return results