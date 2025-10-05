"""
Stellar feature extraction utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class StellarFeatureExtractor:
    """Extract and process stellar parameters for exoplanet detection."""
    
    def __init__(self):
        """Initialize stellar feature extractor."""
        self.config = config_loader.load('config')
        self.feature_config = self.config.get('features', {})
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def extract_basic_features(
        self, 
        stellar_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract basic stellar features.
        
        Args:
            stellar_params: Dictionary with stellar parameters
            
        Returns:
            Dictionary with processed stellar features
        """
        features = {}
        
        # Temperature features
        teff = stellar_params.get('Teff', stellar_params.get('teff', 5778))  # Solar default
        features['teff'] = teff
        features['teff_log'] = np.log10(teff) if teff > 0 else np.log10(5778)
        
        # Stellar class based on temperature
        features['is_hot_star'] = float(teff > 7500)  # A-type and hotter
        features['is_solar_type'] = float(5000 <= teff <= 6500)  # G-type
        features['is_cool_star'] = float(teff < 4000)  # M-type
        
        # Radius features
        radius = stellar_params.get('stellar_radius', stellar_params.get('radius', 1.0))
        features['stellar_radius'] = radius
        features['stellar_radius_log'] = np.log10(radius) if radius > 0 else 0.0
        features['is_giant'] = float(radius > 2.0)
        features['is_dwarf'] = float(radius < 0.8)
        
        # Mass features
        mass = stellar_params.get('stellar_mass', stellar_params.get('mass', 1.0))
        features['stellar_mass'] = mass
        features['stellar_mass_log'] = np.log10(mass) if mass > 0 else 0.0
        
        # Surface gravity
        logg = stellar_params.get('logg', None)
        if logg is None and mass > 0 and radius > 0:
            # Calculate log g from mass and radius
            # log g = log(GM/R^2) = log(M) - 2*log(R) + constant
            # In solar units: log g = log(M/M_sun) - 2*log(R/R_sun) + 4.44
            logg = np.log10(mass) - 2 * np.log10(radius) + 4.44
        
        features['logg'] = logg if logg is not None else 4.44  # Solar default
        features['is_high_gravity'] = float(logg > 4.5) if logg is not None else 0.0
        features['is_low_gravity'] = float(logg < 4.0) if logg is not None else 0.0
        
        # Metallicity features
        metallicity = stellar_params.get('stellar_metallicity', 
                                       stellar_params.get('metallicity', 0.0))
        features['metallicity'] = metallicity
        features['is_metal_rich'] = float(metallicity > 0.2)
        features['is_metal_poor'] = float(metallicity < -0.3)
        
        # Age features (if available)
        age = stellar_params.get('stellar_age', stellar_params.get('age', None))
        if age is not None:
            features['stellar_age'] = age
            features['stellar_age_log'] = np.log10(age) if age > 0 else 0.0
            features['is_young_star'] = float(age < 1.0)  # < 1 Gyr
            features['is_old_star'] = float(age > 10.0)   # > 10 Gyr
        else:
            features['stellar_age'] = 4.6  # Solar age default
            features['stellar_age_log'] = np.log10(4.6)
            features['is_young_star'] = 0.0
            features['is_old_star'] = 0.0
        
        # Magnitude features
        magnitude = stellar_params.get('magnitude', stellar_params.get('mag', 12.0))
        features['magnitude'] = magnitude
        features['is_bright_star'] = float(magnitude < 10.0)
        features['is_faint_star'] = float(magnitude > 15.0)
        
        return features
    
    def calculate_derived_features(
        self, 
        basic_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate derived stellar features.
        
        Args:
            basic_features: Basic stellar features
            
        Returns:
            Dictionary with derived features
        """
        derived = {}
        
        teff = basic_features['teff']
        radius = basic_features['stellar_radius']
        mass = basic_features['stellar_mass']
        logg = basic_features['logg']
        
        # Stellar luminosity (Stefan-Boltzmann law)
        # L = 4π R² σ T⁴, in solar units: L = (R/R_sun)² (T/T_sun)⁴
        t_sun = 5778
        luminosity = (radius ** 2) * ((teff / t_sun) ** 4)
        derived['luminosity'] = luminosity
        derived['luminosity_log'] = np.log10(luminosity) if luminosity > 0 else 0.0
        
        # Stellar density
        if radius > 0:
            density = mass / (radius ** 3)  # In units of solar density
            derived['stellar_density'] = density
            derived['stellar_density_log'] = np.log10(density) if density > 0 else 0.0
        else:
            derived['stellar_density'] = 1.0
            derived['stellar_density_log'] = 0.0
        
        # Main sequence check (using mass-radius relation)
        if mass > 0:
            # Approximate main sequence radius for given mass
            if mass < 0.43:
                expected_radius = 0.8 * (mass ** 0.88)
            elif mass < 2.0:
                expected_radius = mass ** 0.57
            else:
                expected_radius = 1.06 * (mass ** 0.5)
            
            radius_ratio = radius / expected_radius if expected_radius > 0 else 1.0
            derived['radius_deviation'] = abs(np.log10(radius_ratio))
            derived['is_main_sequence'] = float(0.7 < radius_ratio < 1.4)
        else:
            derived['radius_deviation'] = 0.0
            derived['is_main_sequence'] = 1.0
        
        # Color temperature (approximate B-V color)
        if 3500 <= teff <= 8000:
            # Empirical relation for main sequence stars
            bv_color = 0.92 * (5040 / teff - 0.68)
            derived['bv_color'] = bv_color
        else:
            derived['bv_color'] = 0.65  # Solar value
        
        # Spectral type encoding
        if teff >= 30000:
            spectral_type = 0  # O
        elif teff >= 10000:
            spectral_type = 1  # B
        elif teff >= 7500:
            spectral_type = 2  # A
        elif teff >= 6000:
            spectral_type = 3  # F
        elif teff >= 5200:
            spectral_type = 4  # G
        elif teff >= 3700:
            spectral_type = 5  # K
        else:
            spectral_type = 6  # M
        
        derived['spectral_type'] = spectral_type
        
        # Habitable zone features
        if luminosity > 0:
            # Inner and outer edges of habitable zone (in AU)
            hz_inner = np.sqrt(luminosity / 1.1)
            hz_outer = np.sqrt(luminosity / 0.53)
            derived['hz_inner'] = hz_inner
            derived['hz_outer'] = hz_outer
            derived['hz_width'] = hz_outer - hz_inner
        else:
            derived['hz_inner'] = 0.95
            derived['hz_outer'] = 1.37
            derived['hz_width'] = 0.42
        
        return derived
    
    def create_stellar_context_features(
        self, 
        target_features: Dict[str, float],
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """Create features based on stellar population context.
        
        Args:
            target_features: Features for target star
            population_stats: Statistics for stellar population
            
        Returns:
            Context-based features
        """
        context_features = {}
        
        if population_stats is None:
            # Use default stellar population statistics
            population_stats = {
                'teff': {'mean': 5500, 'std': 1000},
                'stellar_radius': {'mean': 1.0, 'std': 0.5},
                'stellar_mass': {'mean': 1.0, 'std': 0.3},
                'metallicity': {'mean': -0.1, 'std': 0.3},
                'magnitude': {'mean': 12.0, 'std': 2.0}
            }
        
        # Calculate z-scores relative to population
        for param, stats in population_stats.items():
            if param in target_features:
                value = target_features[param]
                mean = stats['mean']
                std = stats['std']
                
                if std > 0:
                    z_score = (value - mean) / std
                    context_features[f'{param}_zscore'] = z_score
                    context_features[f'{param}_percentile'] = self._zscore_to_percentile(z_score)
        
        # Identify unusual stars
        teff_z = context_features.get('teff_zscore', 0)
        radius_z = context_features.get('stellar_radius_zscore', 0)
        mass_z = context_features.get('stellar_mass_zscore', 0)
        
        # Overall "unusualness" metric
        unusualness = np.sqrt(teff_z**2 + radius_z**2 + mass_z**2)
        context_features['stellar_unusualness'] = unusualness
        context_features['is_unusual_star'] = float(unusualness > 2.0)
        
        return context_features
    
    def _zscore_to_percentile(self, z_score: float) -> float:
        """Convert z-score to approximate percentile.
        
        Args:
            z_score: Z-score value
            
        Returns:
            Approximate percentile (0-100)
        """
        # Rough approximation using error function
        from math import erf, sqrt
        percentile = 50 * (1 + erf(z_score / sqrt(2)))
        return np.clip(percentile, 0, 100)
    
    def fit_scaler(self, stellar_data: List[Dict[str, float]]) -> None:
        """Fit scaler on stellar feature data.
        
        Args:
            stellar_data: List of stellar feature dictionaries
        """
        if not stellar_data:
            logger.warning("No stellar data provided for scaler fitting")
            return
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(stellar_data)
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_data = df[numeric_cols]
        
        # Fill NaN values with column medians
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        # Fit scaler
        self.scaler.fit(numeric_data)
        self._is_fitted = True
        
        logger.info(f"Fitted scaler on {len(stellar_data)} stellar parameter sets")
    
    def transform_features(
        self, 
        features: Dict[str, float],
        scale: bool = True
    ) -> np.ndarray:
        """Transform stellar features to array format.
        
        Args:
            features: Feature dictionary
            scale: Whether to apply scaling
            
        Returns:
            Feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_array = df[numeric_cols].values[0]
        
        # Fill NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        # Scale if requested and scaler is fitted
        if scale and self._is_fitted:
            feature_array = self.scaler.transform([feature_array])[0]
        
        return feature_array
    
    def extract_all_features(
        self, 
        stellar_params: Dict[str, float],
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """Extract all stellar features.
        
        Args:
            stellar_params: Raw stellar parameters
            population_stats: Population statistics for context
            
        Returns:
            Complete stellar feature dictionary
        """
        try:
            # Extract basic features
            basic_features = self.extract_basic_features(stellar_params)
            
            # Calculate derived features
            derived_features = self.calculate_derived_features(basic_features)
            
            # Create context features
            context_features = self.create_stellar_context_features(
                basic_features, population_stats
            )
            
            # Combine all features
            all_features = {}
            all_features.update(basic_features)
            all_features.update(derived_features)
            all_features.update(context_features)
            
            return all_features
            
        except Exception as e:
            logger.error(f"Stellar feature extraction failed: {e}")
            # Return default features
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default stellar features for failed extractions.
        
        Returns:
            Dictionary with default values
        """
        return {
            'teff': 5778, 'teff_log': np.log10(5778),
            'stellar_radius': 1.0, 'stellar_radius_log': 0.0,
            'stellar_mass': 1.0, 'stellar_mass_log': 0.0,
            'logg': 4.44, 'metallicity': 0.0,
            'stellar_age': 4.6, 'stellar_age_log': np.log10(4.6),
            'magnitude': 12.0, 'luminosity': 1.0, 'luminosity_log': 0.0,
            'stellar_density': 1.0, 'stellar_density_log': 0.0,
            'bv_color': 0.65, 'spectral_type': 4,
            'hz_inner': 0.95, 'hz_outer': 1.37, 'hz_width': 0.42,
            'is_hot_star': 0.0, 'is_solar_type': 1.0, 'is_cool_star': 0.0,
            'is_giant': 0.0, 'is_dwarf': 0.0, 'is_main_sequence': 1.0,
            'is_metal_rich': 0.0, 'is_metal_poor': 0.0,
            'is_young_star': 0.0, 'is_old_star': 0.0,
            'is_bright_star': 0.0, 'is_faint_star': 0.0,
            'radius_deviation': 0.0, 'stellar_unusualness': 0.0,
            'is_unusual_star': 0.0
        }