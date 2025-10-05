"""
Periodogram analysis utilities for exoplanet detection.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.signal import find_peaks
from scipy.stats import binned_statistic

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class PeriodogramAnalyzer:
    """Analyze periodograms for transit detection and characterization."""
    
    def __init__(self):
        """Initialize periodogram analyzer."""
        self.config = config_loader.load('config')
        self.feature_config = self.config.get('features', {})
    
    def box_least_squares(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        period_min: float = 0.5,
        period_max: float = 50.0,
        frequency_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compute Box Least Squares periodogram.
        
        Args:
            time: Time array
            flux: Flux array
            period_min: Minimum period to search
            period_max: Maximum period to search
            frequency_factor: Frequency oversampling factor
            
        Returns:
            Tuple of (periods, power, best_params)
        """
        if len(time) < 10:
            return np.array([]), np.array([]), {}
        
        try:
            from astropy.timeseries import BoxLeastSquares
            
            # Create BLS object
            bls = BoxLeastSquares(time, flux)
            
            # Define period grid
            periods = np.logspace(
                np.log10(period_min),
                np.log10(period_max),
                int(1000 * frequency_factor)
            )
            
            # Compute BLS periodogram
            result = bls.power(periods)
            
            # Find best period
            best_idx = np.argmax(result.power)
            best_period = result.period[best_idx]
            best_power = result.power[best_idx]
            
            # Get additional parameters for best period
            stats = bls.compute_stats(best_period)
            
            best_params = {
                'period': best_period,
                'power': best_power,
                'depth': stats['depth'][0] if len(stats['depth']) > 0 else 0.0,
                'duration': stats['duration'][0] if len(stats['duration']) > 0 else 0.0,
                'transit_time': stats['transit_time'][0] if len(stats['transit_time']) > 0 else 0.0
            }
            
            logger.debug(f"BLS found best period: {best_period:.4f} days (power: {best_power:.4f})")
            
            return result.period, result.power, best_params
            
        except ImportError:
            logger.warning("Astropy not available, using simple periodogram")
            return self._simple_periodogram(time, flux, period_min, period_max)
        except Exception as e:
            logger.error(f"BLS computation failed: {e}")
            return np.array([]), np.array([]), {}
    
    def _simple_periodogram(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        period_min: float = 0.5,
        period_max: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Simple chi-squared periodogram implementation.
        
        Args:
            time: Time array
            flux: Flux array
            period_min: Minimum period
            period_max: Maximum period
            
        Returns:
            Tuple of (periods, power, best_params)
        """
        periods = np.logspace(np.log10(period_min), np.log10(period_max), 500)
        power = np.zeros_like(periods)
        
        baseline = np.median(flux)
        
        for i, period in enumerate(periods):
            # Phase fold
            phase = ((time - time[0]) / period) % 1.0
            phase = np.where(phase > 0.5, phase - 1.0, phase)
            
            # Calculate chi-squared for transit model
            transit_window = 0.1
            transit_mask = np.abs(phase) < (transit_window / 2)
            
            if np.sum(transit_mask) > 1:
                transit_flux = np.mean(flux[transit_mask])
                depth = (baseline - transit_flux) / baseline
                
                # Simple power metric based on depth
                power[i] = depth**2 * np.sum(transit_mask)
        
        # Find best period
        if len(power) > 0:
            best_idx = np.argmax(power)
            best_params = {
                'period': periods[best_idx],
                'power': power[best_idx],
                'depth': 0.0,
                'duration': 0.0,
                'transit_time': 0.0
            }
        else:
            best_params = {}
        
        return periods, power, best_params
    
    def lomb_scargle_periodogram(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        frequency_min: Optional[float] = None,
        frequency_max: Optional[float] = None,
        samples_per_peak: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compute Lomb-Scargle periodogram.
        
        Args:
            time: Time array
            flux: Flux array
            frequency_min: Minimum frequency
            frequency_max: Maximum frequency
            samples_per_peak: Samples per peak
            
        Returns:
            Tuple of (frequencies, power, best_params)
        """
        if len(time) < 10:
            return np.array([]), np.array([]), {}
        
        try:
            from scipy.signal import lombscargle
            
            # Determine frequency range
            if frequency_min is None:
                frequency_min = 1.0 / (np.max(time) - np.min(time))
            if frequency_max is None:
                frequency_max = 0.5 / np.median(np.diff(time))  # Nyquist frequency
            
            # Create frequency grid
            n_frequencies = int(samples_per_peak * (frequency_max - frequency_min) / frequency_min)
            frequencies = np.linspace(frequency_min, frequency_max, n_frequencies)
            
            # Compute periodogram
            power = lombscargle(time, flux - np.mean(flux), frequencies * 2 * np.pi)
            
            # Normalize power
            power = power / np.max(power) if np.max(power) > 0 else power
            
            # Find best frequency
            if len(power) > 0:
                best_idx = np.argmax(power)
                best_frequency = frequencies[best_idx]
                best_period = 1.0 / best_frequency if best_frequency > 0 else 0.0
                
                best_params = {
                    'frequency': best_frequency,
                    'period': best_period,
                    'power': power[best_idx]
                }
            else:
                best_params = {}
            
            logger.debug(f"Lomb-Scargle found best period: {best_params.get('period', 0):.4f} days")
            
            return frequencies, power, best_params
            
        except Exception as e:
            logger.error(f"Lomb-Scargle computation failed: {e}")
            return np.array([]), np.array([]), {}
    
    def detect_periodic_signals(
        self, 
        periods: np.ndarray, 
        power: np.ndarray,
        min_peak_height: float = 0.1,
        min_peak_distance: int = 10
    ) -> List[Dict[str, float]]:
        """Detect periodic signals in periodogram.
        
        Args:
            periods: Period array
            power: Power array
            min_peak_height: Minimum peak height
            min_peak_distance: Minimum distance between peaks
            
        Returns:
            List of detected signals
        """
        if len(power) == 0:
            return []
        
        try:
            # Find peaks in periodogram
            peaks, properties = find_peaks(
                power,
                height=min_peak_height,
                distance=min_peak_distance
            )
            
            signals = []
            for i, peak_idx in enumerate(peaks):
                signal = {
                    'period': periods[peak_idx],
                    'power': power[peak_idx],
                    'rank': i + 1,
                    'prominence': properties.get('prominences', [0])[i] if 'prominences' in properties else 0,
                    'width': properties.get('widths', [0])[i] if 'widths' in properties else 0
                }
                signals.append(signal)
            
            # Sort by power (strongest first)
            signals.sort(key=lambda x: x['power'], reverse=True)
            
            logger.debug(f"Detected {len(signals)} periodic signals")
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal detection failed: {e}")
            return []
    
    def calculate_false_alarm_probability(
        self, 
        power: np.ndarray, 
        target_power: float
    ) -> float:
        """Calculate false alarm probability for given power level.
        
        Args:
            power: Power array from periodogram
            target_power: Target power level
            
        Returns:
            False alarm probability
        """
        if len(power) == 0:
            return 1.0
        
        # Empirical FAP calculation
        n_trials = len(power)
        n_above = np.sum(power >= target_power)
        
        fap = n_above / n_trials
        
        return fap
    
    def extract_periodogram_features(
        self, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, float]:
        """Extract features from periodogram analysis.
        
        Args:
            time: Time array
            flux: Flux array
            
        Returns:
            Dictionary with periodogram features
        """
        features = {}
        
        if len(time) < 10:
            # Return zero features for insufficient data
            feature_names = [
                'bls_period', 'bls_power', 'bls_depth', 'bls_duration',
                'ls_period', 'ls_power', 'n_peaks', 'max_power',
                'power_ratio', 'period_stability'
            ]
            return {name: 0.0 for name in feature_names}
        
        try:
            # BLS periodogram
            bls_periods, bls_power, bls_params = self.box_least_squares(time, flux)
            
            if len(bls_power) > 0:
                features['bls_period'] = bls_params.get('period', 0.0)
                features['bls_power'] = bls_params.get('power', 0.0)
                features['bls_depth'] = bls_params.get('depth', 0.0)
                features['bls_duration'] = bls_params.get('duration', 0.0)
                features['max_power'] = np.max(bls_power)
                
                # Detect multiple peaks
                signals = self.detect_periodic_signals(bls_periods, bls_power)
                features['n_peaks'] = len(signals)
                
                # Power ratio (second highest / highest)
                if len(signals) >= 2:
                    features['power_ratio'] = signals[1]['power'] / signals[0]['power']
                else:
                    features['power_ratio'] = 0.0
            
            else:
                features.update({
                    'bls_period': 0.0, 'bls_power': 0.0, 'bls_depth': 0.0,
                    'bls_duration': 0.0, 'max_power': 0.0, 'n_peaks': 0,
                    'power_ratio': 0.0
                })
            
            # Lomb-Scargle periodogram
            ls_freq, ls_power, ls_params = self.lomb_scargle_periodogram(time, flux)
            
            if len(ls_power) > 0:
                features['ls_period'] = ls_params.get('period', 0.0)
                features['ls_power'] = ls_params.get('power', 0.0)
            else:
                features['ls_period'] = 0.0
                features['ls_power'] = 0.0
            
            # Period stability (consistency between methods)
            if features['bls_period'] > 0 and features['ls_period'] > 0:
                period_diff = abs(features['bls_period'] - features['ls_period'])
                avg_period = (features['bls_period'] + features['ls_period']) / 2
                features['period_stability'] = 1.0 - (period_diff / avg_period)
            else:
                features['period_stability'] = 0.0
            
        except Exception as e:
            logger.error(f"Periodogram feature extraction failed: {e}")
            # Return zero features on failure
            feature_names = [
                'bls_period', 'bls_power', 'bls_depth', 'bls_duration',
                'ls_period', 'ls_power', 'n_peaks', 'max_power',
                'power_ratio', 'period_stability'
            ]
            features = {name: 0.0 for name in feature_names}
        
        return features