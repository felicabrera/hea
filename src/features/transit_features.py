"""
Transit feature extraction utilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.signal import find_peaks

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class TransitFeatureExtractor:
    """Extract transit-related features from light curves."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.config = config_loader.load('config')
        self.feature_config = self.config.get('features', {})
    
    def transit_model(self, phase: np.ndarray, depth: float, duration: float, 
                     baseline: float = 1.0) -> np.ndarray:
        """Simple box transit model.
        
        Args:
            phase: Phase array
            depth: Transit depth
            duration: Transit duration (in phase units)
            baseline: Baseline flux level
            
        Returns:
            Model flux array
        """
        model = np.full_like(phase, baseline)
        transit_mask = np.abs(phase) < (duration / 2)
        model[transit_mask] = baseline * (1 - depth)
        return model
    
    def fit_transit_model(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, float]:
        """Fit simple transit model to phase-folded data.
        
        Args:
            phase: Phase array (centered on transit)
            flux: Flux array
            
        Returns:
            Dictionary with fitted parameters
        """
        try:
            # Initial parameter guesses
            baseline_guess = np.median(flux)
            depth_guess = baseline_guess - np.min(flux)
            duration_guess = 0.1  # 10% of orbit in phase
            
            # Bounds for parameters
            bounds = (
                [0, 0, 0.5],  # Lower bounds: depth, duration, baseline
                [1, 0.5, 2.0]  # Upper bounds
            )
            
            # Fit model
            popt, pcov = curve_fit(
                self.transit_model, phase, flux,
                p0=[depth_guess, duration_guess, baseline_guess],
                bounds=bounds,
                maxfev=1000
            )
            
            depth_fit, duration_fit, baseline_fit = popt
            
            # Calculate goodness of fit
            model_flux = self.transit_model(phase, *popt)
            residuals = flux - model_flux
            chi_squared = np.sum(residuals**2)
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            return {
                'depth': depth_fit,
                'duration': duration_fit,
                'baseline': baseline_fit,
                'depth_error': param_errors[0],
                'duration_error': param_errors[1],
                'baseline_error': param_errors[2],
                'chi_squared': chi_squared,
                'fit_success': True
            }
            
        except Exception as e:
            logger.warning(f"Transit model fitting failed: {e}")
            return {
                'depth': 0.0,
                'duration': 0.0,
                'baseline': 1.0,
                'depth_error': 0.0,
                'duration_error': 0.0,
                'baseline_error': 0.0,
                'chi_squared': np.inf,
                'fit_success': False
            }
    
    def calculate_transit_depth(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray,
        transit_window: float = 0.1
    ) -> float:
        """Calculate transit depth from phase-folded light curve.
        
        Args:
            phase: Phase array
            flux: Flux array
            transit_window: Phase window around transit center
            
        Returns:
            Transit depth
        """
        if len(flux) == 0:
            return 0.0
        
        # Find in-transit points
        transit_mask = np.abs(phase) < (transit_window / 2)
        
        if not np.any(transit_mask):
            return 0.0
        
        # Find out-of-transit points
        out_transit_mask = np.abs(phase) > (transit_window * 1.5)
        
        if not np.any(out_transit_mask):
            baseline = np.median(flux)
        else:
            baseline = np.median(flux[out_transit_mask])
        
        # Calculate depth
        transit_flux = np.median(flux[transit_mask])
        depth = (baseline - transit_flux) / baseline
        
        return max(0.0, depth)  # Ensure non-negative
    
    def calculate_transit_duration(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray,
        depth_threshold: float = 0.5
    ) -> float:
        """Calculate transit duration.
        
        Args:
            phase: Phase array
            flux: Flux array
            depth_threshold: Fraction of full depth to measure duration
            
        Returns:
            Transit duration in phase units
        """
        if len(flux) == 0:
            return 0.0
        
        # Calculate baseline and minimum flux
        baseline = np.median(flux)
        min_flux = np.min(flux)
        
        if baseline <= min_flux:
            return 0.0
        
        # Find threshold flux level
        threshold_flux = baseline - depth_threshold * (baseline - min_flux)
        
        # Find points below threshold
        below_threshold = flux < threshold_flux
        
        if not np.any(below_threshold):
            return 0.0
        
        # Find the extent of the transit
        transit_phases = phase[below_threshold]
        
        if len(transit_phases) == 0:
            return 0.0
        
        duration = np.max(transit_phases) - np.min(transit_phases)
        
        return max(0.0, duration)
    
    def calculate_ingress_egress_duration(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate ingress and egress durations.
        
        Args:
            phase: Phase array
            flux: Flux array
            
        Returns:
            Tuple of (ingress_duration, egress_duration)
        """
        if len(flux) < 10:
            return 0.0, 0.0
        
        try:
            # Sort by phase
            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]
            flux_sorted = flux[sort_idx]
            
            # Find the minimum flux point (transit center)
            min_idx = np.argmin(flux_sorted)
            
            # Split into ingress and egress
            ingress_phase = phase_sorted[:min_idx+1]
            ingress_flux = flux_sorted[:min_idx+1]
            
            egress_phase = phase_sorted[min_idx:]
            egress_flux = flux_sorted[min_idx:]
            
            # Calculate durations based on steepest gradients
            ingress_duration = self._calculate_slope_duration(ingress_phase, ingress_flux)
            egress_duration = self._calculate_slope_duration(egress_phase, egress_flux)
            
            return ingress_duration, egress_duration
            
        except Exception:
            return 0.0, 0.0
    
    def _calculate_slope_duration(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray
    ) -> float:
        """Calculate duration based on steepest slope.
        
        Args:
            phase: Phase array
            flux: Flux array
            
        Returns:
            Duration estimate
        """
        if len(phase) < 3:
            return 0.0
        
        # Calculate gradients
        gradients = np.gradient(flux, phase)
        
        # Find steepest gradients (excluding edges)
        if len(gradients) > 4:
            gradients = gradients[1:-1]
            phase_center = phase[1:-1]
        else:
            phase_center = phase
        
        # Find region with significant gradient
        gradient_threshold = np.std(gradients) * 0.5
        significant_mask = np.abs(gradients) > gradient_threshold
        
        if not np.any(significant_mask):
            return 0.0
        
        significant_phases = phase_center[significant_mask]
        
        if len(significant_phases) == 0:
            return 0.0
        
        duration = np.max(significant_phases) - np.min(significant_phases)
        return max(0.0, duration)
    
    def calculate_transit_snr(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray,
        transit_window: float = 0.1
    ) -> float:
        """Calculate transit signal-to-noise ratio.
        
        Args:
            phase: Phase array
            flux: Flux array
            transit_window: Phase window for transit
            
        Returns:
            Transit SNR
        """
        if len(flux) == 0:
            return 0.0
        
        # Find in-transit and out-of-transit points
        transit_mask = np.abs(phase) < (transit_window / 2)
        out_transit_mask = np.abs(phase) > (transit_window * 1.5)
        
        if not np.any(transit_mask) or not np.any(out_transit_mask):
            return 0.0
        
        # Calculate signal and noise
        transit_flux = np.median(flux[transit_mask])
        baseline_flux = np.median(flux[out_transit_mask])
        noise = np.std(flux[out_transit_mask])
        
        if noise == 0:
            return 0.0
        
        signal = baseline_flux - transit_flux
        snr = signal / noise
        
        return max(0.0, snr)
    
    def calculate_odd_even_difference(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        period: float,
        epoch: float
    ) -> float:
        """Calculate odd-even transit difference.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            epoch: Transit epoch
            
        Returns:
            Odd-even difference metric
        """
        if len(time) == 0 or period <= 0:
            return 0.0
        
        try:
            # Calculate transit numbers
            transit_numbers = np.round((time - epoch) / period)
            
            # Separate odd and even transits
            odd_mask = (transit_numbers % 2) == 1
            even_mask = (transit_numbers % 2) == 0
            
            if not np.any(odd_mask) or not np.any(even_mask):
                return 0.0
            
            # Calculate average depths
            odd_flux = np.median(flux[odd_mask])
            even_flux = np.median(flux[even_mask])
            baseline = np.median(flux)
            
            # Calculate difference
            odd_depth = (baseline - odd_flux) / baseline
            even_depth = (baseline - even_flux) / baseline
            
            odd_even_diff = abs(odd_depth - even_depth)
            
            return odd_even_diff
            
        except Exception:
            return 0.0
    
    def detect_secondary_eclipse(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, float]:
        """Detect and characterize secondary eclipse.
        
        Args:
            phase: Phase array
            flux: Flux array
            
        Returns:
            Dictionary with secondary eclipse properties
        """
        if len(flux) == 0:
            return {'detected': False, 'depth': 0.0, 'phase': 0.5}
        
        # Search for secondary eclipse around phase 0.5
        search_window = 0.2
        search_center = 0.5
        
        # Adjust phase for search (secondary eclipse at phase 0.5)
        adjusted_phase = phase.copy()
        adjusted_phase[adjusted_phase < 0] += 1.0
        
        search_mask = np.abs(adjusted_phase - search_center) < (search_window / 2)
        
        if not np.any(search_mask):
            return {'detected': False, 'depth': 0.0, 'phase': search_center}
        
        # Find minimum in search window
        search_flux = flux[search_mask]
        search_phase = adjusted_phase[search_mask]
        
        min_idx = np.argmin(search_flux)
        secondary_phase = search_phase[min_idx]
        secondary_flux = search_flux[min_idx]
        
        # Calculate baseline from out-of-eclipse regions
        out_eclipse_mask = (np.abs(adjusted_phase - search_center) > search_window) & \
                          (np.abs(phase) > 0.2)  # Also avoid primary eclipse
        
        if np.any(out_eclipse_mask):
            baseline = np.median(flux[out_eclipse_mask])
        else:
            baseline = np.median(flux)
        
        # Calculate depth and significance
        secondary_depth = (baseline - secondary_flux) / baseline
        noise_level = np.std(flux[out_eclipse_mask]) if np.any(out_eclipse_mask) else np.std(flux)
        
        # Detection threshold (3-sigma)
        detection_threshold = 3 * noise_level / baseline
        detected = secondary_depth > detection_threshold
        
        return {
            'detected': detected,
            'depth': max(0.0, secondary_depth),
            'phase': secondary_phase,
            'significance': secondary_depth / (noise_level / baseline) if noise_level > 0 else 0.0
        }
    
    def extract_all_features(
        self, 
        phase: np.ndarray, 
        flux: np.ndarray,
        time: Optional[np.ndarray] = None,
        period: Optional[float] = None,
        epoch: Optional[float] = None
    ) -> Dict[str, float]:
        """Extract all transit features.
        
        Args:
            phase: Phase array
            flux: Flux array
            time: Original time array (optional)
            period: Orbital period (optional)
            epoch: Transit epoch (optional)
            
        Returns:
            Dictionary with all transit features
        """
        features = {}
        
        if len(flux) == 0:
            # Return zero features for empty data
            feature_names = [
                'depth', 'duration', 'snr', 'ingress_duration', 'egress_duration',
                'secondary_depth', 'secondary_detected', 'odd_even_diff',
                 'baseline_flux', 'min_flux', 'transit_asymmetry'
            ]
            return {name: 0.0 for name in feature_names}
        
        try:
            # Basic transit features
            features['depth'] = self.calculate_transit_depth(phase, flux)
            features['duration'] = self.calculate_transit_duration(phase, flux)
            features['snr'] = self.calculate_transit_snr(phase, flux)
            
            # Ingress/egress features
            ingress_dur, egress_dur = self.calculate_ingress_egress_duration(phase, flux)
            features['ingress_duration'] = ingress_dur
            features['egress_duration'] = egress_dur
            
            # Asymmetry measure
            features['transit_asymmetry'] = abs(ingress_dur - egress_dur)
            
            # Secondary eclipse
            secondary_info = self.detect_secondary_eclipse(phase, flux)
            features['secondary_depth'] = secondary_info['depth']
            features['secondary_detected'] = float(secondary_info['detected'])
            
            # Odd-even difference (if time series provided)
            if time is not None and period is not None and epoch is not None:
                features['odd_even_diff'] = self.calculate_odd_even_difference(
                    time, flux, period, epoch
                )
            else:
                features['odd_even_diff'] = 0.0
            
            # Basic flux statistics
            features['baseline_flux'] = np.median(flux)
            features['min_flux'] = np.min(flux)
            features['flux_std'] = np.std(flux)
            
            # Transit model fit
            model_params = self.fit_transit_model(phase, flux)
            features['model_depth'] = model_params['depth']
            features['model_duration'] = model_params['duration']
            features['model_chi2'] = model_params['chi_squared']
            features['model_fit_success'] = float(model_params['fit_success'])
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Return zero features on failure
            feature_names = [
                'depth', 'duration', 'snr', 'ingress_duration', 'egress_duration',
                'secondary_depth', 'secondary_detected', 'odd_even_diff',
                'baseline_flux', 'min_flux', 'flux_std', 'transit_asymmetry',
                'model_depth', 'model_duration', 'model_chi2', 'model_fit_success'
            ]
            features = {name: 0.0 for name in feature_names}
        
        return features