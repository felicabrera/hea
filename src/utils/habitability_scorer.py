"""
Habitability Scoring System for Exoplanet Detection
NASA Space Apps Challenge 2025

Calculates habitability scores based on:
- Planet size (Earth-like radius)
- Orbital period (habitable zone)
- Equilibrium temperature
- Stellar properties
- Insolation flux

References:
- Kopparapu et al. (2013) - Habitable Zone boundaries
- Kane et al. (2016) - Earth Similarity Index
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class HabitabilityScorer:
    """
    Calculate habitability scores for exoplanet candidates.
    
    Scoring Components:
    1. Radius Score: How Earth-like is the planet size?
    2. Temperature Score: Is it in the habitable temperature range?
    3. Insolation Score: Does it receive Earth-like stellar flux?
    4. Stellar Score: Is the host star suitable for life?
    5. Orbital Score: Is the orbital period in the habitable range?
    """
    
    # Constants (Earth reference values)
    EARTH_RADIUS = 1.0  # Earth radii
    EARTH_TEMP_MIN = 273.15  # 0°C in Kelvin (minimum for liquid water)
    EARTH_TEMP_MAX = 373.15  # 100°C in Kelvin (maximum for liquid water)
    EARTH_TEMP_OPTIMAL = 288.0  # ~15°C (Earth's average)
    EARTH_INSOLATION = 1.0  # Earth flux
    SUN_TEMP = 5778.0  # Kelvin
    
    # Habitable Zone boundaries (conservative)
    HZ_INNER_FLUX = 1.776  # Too hot (runaway greenhouse)
    HZ_OUTER_FLUX = 0.32   # Too cold (maximum greenhouse)
    
    # Optimistic HZ boundaries
    HZ_INNER_FLUX_OPT = 1.107  # Recent Venus
    HZ_OUTER_FLUX_OPT = 0.356  # Early Mars
    
    # Planet radius categories (Fulton gap)
    ROCKY_MAX_RADIUS = 1.8  # Earth radii
    SUPER_EARTH_MAX = 2.5
    MINI_NEPTUNE_MAX = 4.0
    
    # Stellar type preferences (Main sequence)
    STELLAR_TEMP_MIN = 3500  # M-dwarf minimum
    STELLAR_TEMP_MAX = 7000  # F-star maximum
    STELLAR_TEMP_OPTIMAL_MIN = 4500  # K-star
    STELLAR_TEMP_OPTIMAL_MAX = 6500  # G-star
    
    def __init__(self, scoring_mode: str = 'balanced'):
        """
        Initialize habitability scorer.
        
        Args:
            scoring_mode: 'conservative', 'optimistic', or 'balanced'
        """
        self.scoring_mode = scoring_mode
        
        if scoring_mode == 'conservative':
            self.hz_inner = self.HZ_INNER_FLUX
            self.hz_outer = self.HZ_OUTER_FLUX
            self.temp_range = (self.EARTH_TEMP_MIN, self.EARTH_TEMP_MAX)
        elif scoring_mode == 'optimistic':
            self.hz_inner = self.HZ_INNER_FLUX_OPT
            self.hz_outer = self.HZ_OUTER_FLUX_OPT
            self.temp_range = (200, 400)  # Extended range
        else:  # balanced
            self.hz_inner = 1.4
            self.hz_outer = 0.35
            self.temp_range = (250, 350)
    
    def calculate_radius_score(self, radius: float) -> float:
        """
        Score based on planet radius (Earth-like preferred).
        
        Scoring:
        - 1.0: Earth-sized (0.8-1.2 R⊕)
        - 0.8-0.9: Rocky super-Earth (1.2-1.8 R⊕)
        - 0.5-0.7: Large rocky (1.8-2.5 R⊕)
        - 0.1-0.3: Mini-Neptune (2.5-4.0 R⊕)
        - 0.0: Neptune-sized or larger (>4.0 R⊕)
        """
        if pd.isna(radius):
            return np.nan
        
        if 0.8 <= radius <= 1.2:
            # Earth-sized - perfect!
            return 1.0
        elif 1.2 < radius <= self.ROCKY_MAX_RADIUS:
            # Super-Earth - likely rocky
            return 0.9 - 0.1 * (radius - 1.2) / (self.ROCKY_MAX_RADIUS - 1.2)
        elif 0.5 <= radius < 0.8:
            # Mars/Venus sized
            return 0.9 - 0.2 * (0.8 - radius) / 0.3
        elif self.ROCKY_MAX_RADIUS < radius <= self.SUPER_EARTH_MAX:
            # Large super-Earth - might retain H/He
            return 0.7 - 0.2 * (radius - self.ROCKY_MAX_RADIUS) / (self.SUPER_EARTH_MAX - self.ROCKY_MAX_RADIUS)
        elif self.SUPER_EARTH_MAX < radius <= self.MINI_NEPTUNE_MAX:
            # Mini-Neptune - unlikely habitable
            return 0.3 - 0.2 * (radius - self.SUPER_EARTH_MAX) / (self.MINI_NEPTUNE_MAX - self.SUPER_EARTH_MAX)
        else:
            # Too large (gas giant) or too small
            return 0.1 if radius > self.MINI_NEPTUNE_MAX else 0.3
    
    def calculate_temperature_score(self, temp: float) -> float:
        """
        Score based on equilibrium temperature.
        
        Scoring:
        - 1.0: Earth-like (260-320 K)
        - 0.8-0.9: Temperate (230-350 K)
        - 0.4-0.6: Cold or warm (180-380 K)
        - 0.0-0.2: Extreme temperatures
        """
        if pd.isna(temp):
            return np.nan
        
        temp_min, temp_max = self.temp_range
        
        if 260 <= temp <= 320:
            # Earth-like temperature
            return 1.0
        elif temp_min <= temp < 260:
            # Cool but possibly habitable
            return 0.8 + 0.2 * (temp - temp_min) / (260 - temp_min)
        elif 320 < temp <= temp_max:
            # Warm but possibly habitable
            return 0.8 + 0.2 * (temp_max - temp) / (temp_max - 320)
        elif temp < temp_min or temp > temp_max:
            # Outside habitable range
            distance = min(abs(temp - temp_min), abs(temp - temp_max))
            return max(0.0, 0.4 - distance / 200)
        else:
            return 0.5
    
    def calculate_insolation_score(self, insolation: float) -> float:
        """
        Score based on stellar flux received (Earth flux = 1.0).
        
        Scoring:
        - 1.0: Earth-like (0.9-1.1 S⊕)
        - 0.8-0.9: In habitable zone
        - 0.3-0.6: Near HZ boundaries
        - 0.0: Outside habitable zone
        """
        if pd.isna(insolation):
            return np.nan
        
        if 0.9 <= insolation <= 1.1:
            # Earth-like insolation
            return 1.0
        elif self.hz_outer <= insolation < 0.9:
            # Cool but in HZ
            return 0.85 + 0.15 * (insolation - self.hz_outer) / (0.9 - self.hz_outer)
        elif 1.1 < insolation <= self.hz_inner:
            # Warm but in HZ
            return 0.85 + 0.15 * (self.hz_inner - insolation) / (self.hz_inner - 1.1)
        elif insolation < self.hz_outer:
            # Too cold (outside HZ)
            return max(0.0, 0.3 - (self.hz_outer - insolation) / 2)
        elif insolation > self.hz_inner:
            # Too hot (outside HZ)
            return max(0.0, 0.3 - (insolation - self.hz_inner) / 5)
        else:
            return 0.5
    
    def calculate_stellar_score(self, stellar_temp: float) -> float:
        """
        Score based on host star temperature.
        
        Stellar types and habitability:
        - G-type (Sun-like): 5200-6000 K - Best
        - K-type (Orange dwarf): 3900-5200 K - Good
        - F-type (Yellow-white): 6000-7500 K - Acceptable
        - M-type (Red dwarf): 2500-3900 K - Debated (tidal locking, flares)
        """
        if pd.isna(stellar_temp):
            return np.nan
        
        if self.STELLAR_TEMP_OPTIMAL_MIN <= stellar_temp <= self.STELLAR_TEMP_OPTIMAL_MAX:
            # G/K type stars - ideal for life
            return 1.0
        elif 3900 <= stellar_temp < self.STELLAR_TEMP_OPTIMAL_MIN:
            # K/M boundary - good but cooler
            return 0.85
        elif self.STELLAR_TEMP_OPTIMAL_MAX < stellar_temp <= 7000:
            # Early F-type - hotter but acceptable
            return 0.8 - 0.2 * (stellar_temp - self.STELLAR_TEMP_OPTIMAL_MAX) / (7000 - self.STELLAR_TEMP_OPTIMAL_MAX)
        elif self.STELLAR_TEMP_MIN <= stellar_temp < 3900:
            # M-dwarfs - controversial
            return 0.6 - 0.2 * (3900 - stellar_temp) / (3900 - self.STELLAR_TEMP_MIN)
        else:
            # Too hot (A-type) or too cool
            return 0.2
    
    def calculate_orbital_score(self, period: float, stellar_temp: Optional[float] = None) -> float:
        """
        Score based on orbital period.
        
        Rough habitable zone periods (depends on stellar type):
        - G-type star: 200-600 days
        - K-type star: 100-400 days
        - M-type star: 10-100 days
        """
        if pd.isna(period):
            return np.nan
        
        # Estimate expected HZ period range based on stellar type
        if pd.notna(stellar_temp):
            if stellar_temp >= 5200:  # G-type or hotter
                hz_min, hz_max = 200, 600
            elif stellar_temp >= 3900:  # K-type
                hz_min, hz_max = 100, 400
            else:  # M-type
                hz_min, hz_max = 10, 100
        else:
            # Unknown star type - use broad range
            hz_min, hz_max = 50, 500
        
        if hz_min <= period <= hz_max:
            # In expected HZ period range
            center = (hz_min + hz_max) / 2
            return 1.0 - 0.3 * abs(period - center) / center
        elif period < hz_min:
            # Too close (hot)
            return max(0.0, 0.6 - (hz_min - period) / hz_min)
        else:
            # Too far (cold)
            return max(0.0, 0.6 - (period - hz_max) / hz_max)
    
    def calculate_earth_similarity_index(self, radius: float, temp: float) -> float:
        """
        Calculate Earth Similarity Index (ESI).
        
        ESI = Product of weighted geometric means of:
        - Radius ratio
        - Temperature ratio
        
        Range: 0.0 (not similar) to 1.0 (Earth twin)
        """
        if pd.isna(radius) or pd.isna(temp):
            return np.nan
        
        # Radius component
        radius_ratio = min(radius / self.EARTH_RADIUS, self.EARTH_RADIUS / radius)
        radius_component = (1 - abs((radius - self.EARTH_RADIUS) / (radius + self.EARTH_RADIUS))) ** 0.5
        
        # Temperature component  
        temp_ratio = min(temp / self.EARTH_TEMP_OPTIMAL, self.EARTH_TEMP_OPTIMAL / temp)
        temp_component = (1 - abs((temp - self.EARTH_TEMP_OPTIMAL) / (temp + self.EARTH_TEMP_OPTIMAL))) ** 0.5
        
        # Geometric mean
        esi = (radius_component * temp_component) ** 0.5
        return esi
    
    def calculate_habitability_score(
        self,
        radius: Optional[float] = None,
        temp: Optional[float] = None,
        insolation: Optional[float] = None,
        stellar_temp: Optional[float] = None,
        period: Optional[float] = None,
        disposition: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive habitability score.
        
        Returns:
            Dictionary with individual scores and overall habitability score
        """
        scores = {}
        
        # Calculate individual scores
        scores['radius_score'] = self.calculate_radius_score(radius)
        scores['temperature_score'] = self.calculate_temperature_score(temp)
        scores['insolation_score'] = self.calculate_insolation_score(insolation)
        scores['stellar_score'] = self.calculate_stellar_score(stellar_temp)
        scores['orbital_score'] = self.calculate_orbital_score(period, stellar_temp)
        scores['esi'] = self.calculate_earth_similarity_index(radius, temp)
        
        # Calculate overall score (weighted average of available scores)
        weights = {
            'radius_score': 0.25,
            'temperature_score': 0.25,
            'insolation_score': 0.20,
            'stellar_score': 0.15,
            'orbital_score': 0.15
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for key, weight in weights.items():
            if pd.notna(scores[key]):
                weighted_sum += scores[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            scores['habitability_score'] = weighted_sum / total_weight
        else:
            scores['habitability_score'] = np.nan
        
        # Disposition bonus (confirmed planets get slight boost)
        if disposition == 'CONFIRMED' and pd.notna(scores['habitability_score']):
            scores['habitability_score'] = min(1.0, scores['habitability_score'] * 1.05)
        
        # Habitability classification
        if pd.notna(scores['habitability_score']):
            if scores['habitability_score'] >= 0.8:
                scores['habitability_class'] = 'HIGH'
            elif scores['habitability_score'] >= 0.6:
                scores['habitability_class'] = 'MODERATE'
            elif scores['habitability_score'] >= 0.4:
                scores['habitability_class'] = 'LOW'
            else:
                scores['habitability_class'] = 'VERY_LOW'
        else:
            scores['habitability_class'] = 'UNKNOWN'
        
        return scores
    
    def score_dataset(self, df: pd.DataFrame, mission: str = 'kepler') -> pd.DataFrame:
        """
        Score an entire dataset of exoplanet candidates.
        
        Args:
            df: DataFrame with exoplanet data
            mission: 'kepler', 'tess', or 'k2' (determines column names)
        
        Returns:
            DataFrame with added habitability scores
        """
        df_scored = df.copy()
        
        # Map mission-specific column names
        if mission.lower() == 'kepler':
            col_map = {
                'radius': 'koi_prad',
                'temp': 'koi_teq',
                'insolation': 'koi_insol',
                'stellar_temp': 'koi_steff',
                'period': 'koi_period',
                'disposition': 'koi_disposition'
            }
        elif mission.lower() == 'tess':
            col_map = {
                'radius': 'pl_rade',
                'temp': 'pl_eqt',
                'insolation': 'pl_insol',
                'stellar_temp': 'st_teff',
                'period': 'pl_orbper',
                'disposition': 'tfopwg_disp'
            }
        elif mission.lower() == 'k2':
            col_map = {
                'radius': 'pl_rade',
                'temp': 'pl_eqt',
                'insolation': 'pl_insol',
                'stellar_temp': 'st_teff',
                'period': 'pl_orbper',
                'disposition': 'disposition'
            }
        else:
            raise ValueError(f"Unknown mission: {mission}")
        
        # Calculate scores for each row
        scores_list = []
        for idx, row in df_scored.iterrows():
            kwargs = {}
            for key, col in col_map.items():
                if col in df_scored.columns:
                    kwargs[key] = row[col]
            
            scores = self.calculate_habitability_score(**kwargs)
            scores_list.append(scores)
        
        # Add scores to dataframe
        scores_df = pd.DataFrame(scores_list)
        for col in scores_df.columns:
            df_scored[f'hab_{col}'] = scores_df[col]
        
        return df_scored
    
    def get_top_habitable_candidates(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
        min_score: float = 0.6
    ) -> pd.DataFrame:
        """
        Get top potentially habitable candidates.
        
        Args:
            df: Scored DataFrame
            top_n: Number of top candidates to return
            min_score: Minimum habitability score threshold
        
        Returns:
            DataFrame with top candidates sorted by habitability score
        """
        if 'hab_habitability_score' not in df.columns:
            raise ValueError("Dataset must be scored first. Call score_dataset().")
        
        # Filter by minimum score and sort
        top_candidates = df[
            df['hab_habitability_score'] >= min_score
        ].sort_values('hab_habitability_score', ascending=False).head(top_n)
        
        return top_candidates


def demonstrate_habitability_scoring():
    """
    Demonstration of habitability scoring system.
    """
    print("=" * 80)
    print("HABITABILITY SCORING SYSTEM - DEMONSTRATION")
    print("=" * 80)
    
    # Example planets
    examples = [
        {
            'name': 'Earth (Reference)',
            'radius': 1.0,
            'temp': 288,
            'insolation': 1.0,
            'stellar_temp': 5778,
            'period': 365.25
        },
        {
            'name': 'Mars-like',
            'radius': 0.53,
            'temp': 210,
            'insolation': 0.43,
            'stellar_temp': 5778,
            'period': 687
        },
        {
            'name': 'Venus-like',
            'radius': 0.95,
            'temp': 737,
            'insolation': 1.91,
            'stellar_temp': 5778,
            'period': 225
        },
        {
            'name': 'Super-Earth (Habitable)',
            'radius': 1.6,
            'temp': 280,
            'insolation': 1.1,
            'stellar_temp': 5200,
            'period': 200
        },
        {
            'name': 'Hot Jupiter',
            'radius': 11.0,
            'temp': 1500,
            'insolation': 2000,
            'stellar_temp': 6200,
            'period': 3.5
        }
    ]
    
    scorer = HabitabilityScorer(scoring_mode='balanced')
    
    for example in examples:
        name = example.pop('name')
        scores = scorer.calculate_habitability_score(**example)
        
        print(f"\n{name}:")
        print(f"  Overall Habitability Score: {scores['habitability_score']:.3f}")
        print(f"  Classification: {scores['habitability_class']}")
        print(f"  ESI: {scores['esi']:.3f}" if pd.notna(scores['esi']) else "  ESI: N/A")
        print(f"  Component Scores:")
        print(f"    - Radius: {scores['radius_score']:.3f}")
        print(f"    - Temperature: {scores['temperature_score']:.3f}")
        print(f"    - Insolation: {scores['insolation_score']:.3f}")
        print(f"    - Stellar: {scores['stellar_score']:.3f}")
        print(f"    - Orbital: {scores['orbital_score']:.3f}")


if __name__ == '__main__':
    demonstrate_habitability_scoring()
