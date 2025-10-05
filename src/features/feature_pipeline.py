"""
Complete feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .transit_features import TransitFeatureExtractor
from .stellar_features import StellarFeatureExtractor
from .periodogram import PeriodogramAnalyzer
from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class FeaturePipeline:
    """Complete feature engineering pipeline for exoplanet detection."""
    
    def __init__(self):
        """Initialize feature pipeline."""
        self.config = config_loader.load('config')
        self.feature_config = self.config.get('features', {})
        
        # Initialize extractors
        self.transit_extractor = TransitFeatureExtractor()
        self.stellar_extractor = StellarFeatureExtractor()
        self.periodogram_analyzer = PeriodogramAnalyzer()
        
        # Feature names for consistent ordering
        self.feature_names = []
        self._feature_names_initialized = False
    
    def extract_light_curve_features(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        phase: Optional[np.ndarray] = None,
        period: Optional[float] = None,
        epoch: Optional[float] = None
    ) -> Dict[str, float]:
        """Extract all light curve features.
        
        Args:
            time: Time array
            flux: Flux array
            phase: Phase array (optional)
            period: Orbital period
            epoch: Transit epoch
            
        Returns:
            Dictionary with all light curve features
        """
        features = {}
        
        # Transit features (require phase-folded data)
        if phase is not None and len(phase) > 0:
            transit_features = self.transit_extractor.extract_all_features(
                phase, flux, time, period, epoch
            )
            features.update(transit_features)
        
        # Periodogram features (require time series)
        if len(time) > 10:
            use_periodogram = self.feature_config.get('use_periodogram', True)
            if use_periodogram:
                periodogram_features = self.periodogram_analyzer.extract_periodogram_features(
                    time, flux
                )
                features.update(periodogram_features)
        
        # Basic statistical features
        stats_features = self._extract_statistical_features(flux)
        features.update(stats_features)
        
        return features
    
    def extract_stellar_features(
        self, 
        stellar_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract stellar features.
        
        Args:
            stellar_params: Raw stellar parameters
            
        Returns:
            Dictionary with stellar features
        """
        use_stellar = self.feature_config.get('use_stellar_params', True)
        
        if not use_stellar:
            return {}
        
        return self.stellar_extractor.extract_all_features(stellar_params)
    
    def _extract_statistical_features(self, flux: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features from flux.
        
        Args:
            flux: Flux array
            
        Returns:
            Statistical features
        """
        if len(flux) == 0:
            return {
                'flux_mean': 0.0, 'flux_std': 0.0, 'flux_var': 0.0,
                'flux_skew': 0.0, 'flux_kurt': 0.0, 'flux_range': 0.0,
                'flux_iqr': 0.0, 'flux_mad': 0.0
            }
        
        try:
            from scipy import stats
            
            features = {
                'flux_mean': np.mean(flux),
                'flux_std': np.std(flux),
                'flux_var': np.var(flux),
                'flux_median': np.median(flux),
                'flux_min': np.min(flux),
                'flux_max': np.max(flux),
                'flux_range': np.max(flux) - np.min(flux),
                'flux_skew': stats.skew(flux),
                'flux_kurt': stats.kurtosis(flux)
            }
            
            # Interquartile range
            q25, q75 = np.percentile(flux, [25, 75])
            features['flux_iqr'] = q75 - q25
            
            # Median absolute deviation
            features['flux_mad'] = np.median(np.abs(flux - np.median(flux)))
            
            return features
            
        except Exception as e:
            logger.warning(f"Statistical feature extraction failed: {e}")
            return {
                'flux_mean': np.mean(flux) if len(flux) > 0 else 0.0,
                'flux_std': np.std(flux) if len(flux) > 0 else 0.0,
                'flux_var': 0.0, 'flux_skew': 0.0, 'flux_kurt': 0.0,
                'flux_range': 0.0, 'flux_iqr': 0.0, 'flux_mad': 0.0,
                'flux_median': 0.0, 'flux_min': 0.0, 'flux_max': 0.0
            }
    
    def combine_features(
        self, 
        light_curve_features: Dict[str, float],
        stellar_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine light curve and stellar features.
        
        Args:
            light_curve_features: Light curve features
            stellar_features: Stellar features
            
        Returns:
            Combined feature dictionary
        """
        combined = {}
        combined.update(light_curve_features)
        combined.update(stellar_features)
        
        # Create interaction features
        interaction_features = self._create_interaction_features(
            light_curve_features, stellar_features
        )
        combined.update(interaction_features)
        
        return combined
    
    def _create_interaction_features(
        self, 
        lc_features: Dict[str, float],
        stellar_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Create interaction features between light curve and stellar parameters.
        
        Args:
            lc_features: Light curve features
            stellar_features: Stellar features
            
        Returns:
            Interaction features
        """
        interactions = {}
        
        try:
            # Transit depth vs stellar radius (expected scaling)
            if 'depth' in lc_features and 'stellar_radius' in stellar_features:
                # Smaller stars should show deeper transits for same planet
                depth = lc_features['depth']
                radius = stellar_features['stellar_radius']
                if radius > 0:
                    interactions['depth_radius_ratio'] = depth * (radius ** 2)
            
            # SNR vs stellar magnitude (brightness)
            if 'snr' in lc_features and 'magnitude' in stellar_features:
                snr = lc_features['snr']
                mag = stellar_features['magnitude']
                # Brighter stars (lower magnitude) should have higher SNR
                interactions['snr_brightness_consistency'] = snr * np.exp(-0.4 * (mag - 10))
            
            # Duration vs period consistency
            if 'duration' in lc_features and 'bls_period' in lc_features:
                duration = lc_features['duration']
                period = lc_features['bls_period']
                if period > 0:
                    interactions['duration_period_ratio'] = duration / period
            
            # Stellar density vs transit duration
            if 'duration' in lc_features and 'stellar_density' in stellar_features:
                duration = lc_features['duration']
                density = stellar_features['stellar_density']
                # Higher density should correlate with shorter transits
                interactions['duration_density_product'] = duration * np.sqrt(density)
            
        except Exception as e:
            logger.warning(f"Interaction feature creation failed: {e}")
        
        return interactions
    
    def process_single_target(
        self, 
        time: np.ndarray,
        flux: np.ndarray,
        stellar_params: Dict[str, float],
        phase: Optional[np.ndarray] = None,
        period: Optional[float] = None,
        epoch: Optional[float] = None,
        target_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Process a single target through the complete feature pipeline.
        
        Args:
            time: Time array
            flux: Flux array
            stellar_params: Stellar parameters
            phase: Phase array (optional)
            period: Orbital period
            epoch: Transit epoch
            target_id: Target identifier
            
        Returns:
            Complete feature dictionary
        """
        if target_id is None:
            target_id = "unknown"
        
        try:
            # Extract light curve features
            lc_features = self.extract_light_curve_features(
                time, flux, phase, period, epoch
            )
            
            # Extract stellar features
            stellar_features = self.extract_stellar_features(stellar_params)
            
            # Combine features
            all_features = self.combine_features(lc_features, stellar_features)
            
            # Initialize feature names if first time
            if not self._feature_names_initialized:
                self.feature_names = sorted(all_features.keys())
                self._feature_names_initialized = True
                logger.info(f"Initialized {len(self.feature_names)} feature names")
            
            logger.debug(f"[{target_id}] Extracted {len(all_features)} features")
            
            return all_features
            
        except Exception as e:
            logger.error(f"[{target_id}] Feature extraction failed: {e}")
            return self._get_default_features()
    
    def process_batch(
        self, 
        time_arrays: List[np.ndarray],
        flux_arrays: List[np.ndarray],
        stellar_params_list: List[Dict[str, float]],
        phase_arrays: Optional[List[np.ndarray]] = None,
        periods: Optional[List[float]] = None,
        epochs: Optional[List[float]] = None,
        target_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process a batch of targets.
        
        Args:
            time_arrays: List of time arrays
            flux_arrays: List of flux arrays
            stellar_params_list: List of stellar parameter dictionaries
            phase_arrays: List of phase arrays (optional)
            periods: List of periods
            epochs: List of epochs
            target_ids: List of target IDs
            
        Returns:
            DataFrame with features for all targets
        """
        n_targets = len(flux_arrays)
        
        # Set defaults
        if phase_arrays is None:
            phase_arrays = [None] * n_targets
        if periods is None:
            periods = [None] * n_targets
        if epochs is None:
            epochs = [None] * n_targets
        if target_ids is None:
            target_ids = [f"target_{i}" for i in range(n_targets)]
        
        features_list = []
        
        logger.info(f"Processing features for {n_targets} targets...")
        
        for i in range(n_targets):
            if i % 100 == 0:
                logger.info(f"Processing target {i+1}/{n_targets}")
            
            features = self.process_single_target(
                time_arrays[i],
                flux_arrays[i],
                stellar_params_list[i],
                phase_arrays[i],
                periods[i],
                epochs[i],
                target_ids[i]
            )
            
            features['target_id'] = target_ids[i]
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Ensure consistent column order
        if self.feature_names:
            feature_cols = [col for col in self.feature_names if col in df.columns]
            other_cols = [col for col in df.columns if col not in feature_cols]
            df = df[other_cols + feature_cols]
        
        logger.info(f"Feature extraction completed: {len(df)} targets, {len(df.columns)} features")
        
        return df
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features for failed extractions.
        
        Returns:
            Dictionary with default feature values
        """
        if self.feature_names:
            return {name: 0.0 for name in self.feature_names}
        else:
            # Minimal default features
            return {
                'depth': 0.0, 'duration': 0.0, 'snr': 0.0,
                'flux_mean': 1.0, 'flux_std': 0.0,
                'teff': 5778, 'stellar_radius': 1.0, 'stellar_mass': 1.0
            }
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy() if self.feature_names else []
    
    def save_feature_config(self, filepath: Path) -> None:
        """Save feature configuration and names.
        
        Args:
            filepath: Path to save configuration
        """
        import json
        
        config = {
            'feature_names': self.feature_names,
            'feature_config': self.feature_config,
            'n_features': len(self.feature_names)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved feature configuration to {filepath}")
    
    def load_feature_config(self, filepath: Path) -> None:
        """Load feature configuration and names.
        
        Args:
            filepath: Path to load configuration from
        """
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.feature_names = config['feature_names']
        self._feature_names_initialized = True
        
        logger.info(f"Loaded feature configuration from {filepath}")
        logger.info(f"Initialized {len(self.feature_names)} feature names")