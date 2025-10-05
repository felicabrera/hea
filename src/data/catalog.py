"""
Catalog management utilities for exoplanet data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class CatalogManager:
    """Manage exoplanet catalogs and labels."""
    
    def __init__(self):
        """Initialize catalog manager."""
        self.config = config_loader.load('config')
        self.catalog_dir = config_loader.get_path(self.config['data']['catalog_dir'])
        
        self._catalogs = {}
        self._labels = {}
    
    def load_catalog(self, mission: str) -> pd.DataFrame:
        """Load catalog for a specific mission.
        
        Args:
            mission: Mission name ('kepler', 'tess', 'k2')
            
        Returns:
            Catalog DataFrame
        """
        if mission in self._catalogs:
            return self._catalogs[mission]
        
        # Determine catalog filename
        if mission.lower() == 'kepler':
            filename = 'kepler_koi.csv'
        elif mission.lower() == 'tess':
            filename = 'tess_toi.csv'
        elif mission.lower() == 'k2':
            filename = 'k2_catalog.csv'
        else:
            raise ValueError(f"Unknown mission: {mission}")
        
        catalog_path = self.catalog_dir / filename
        
        if not catalog_path.exists():
            download_urls = {
                'kepler_koi.csv': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi',
                'tess_toi.csv': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=toi',
                'k2_catalog.csv': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates'
            }
            url = download_urls.get(filename, "NASA Exoplanet Archive")
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}\nPlease download from: {url}")
        
        # NASA Exoplanet Archive CSV files have comment headers starting with #
        catalog = pd.read_csv(catalog_path, comment='#')
        self._catalogs[mission] = catalog
        
        logger.info(f"Loaded {len(catalog)} entries from {mission} catalog ({catalog_path})")
        return catalog
    
    def get_labels(self, mission: str) -> Dict[str, int]:
        """Get binary labels for exoplanet classification.
        
        Args:
            mission: Mission name
            
        Returns:
            Dictionary mapping target_id to label (0=false positive, 1=exoplanet)
        """
        if mission in self._labels:
            return self._labels[mission]
        
        catalog = self.load_catalog(mission)
        labels = {}
        
        if mission.lower() == 'kepler':
            # Map KOI disposition to binary labels
            for _, row in catalog.iterrows():
                target_id = f"KIC {int(row['kepid'])}"
                
                if row['koi_disposition'] == 'CONFIRMED':
                    labels[target_id] = 1  # Exoplanet
                elif row['koi_disposition'] == 'FALSE POSITIVE':
                    labels[target_id] = 0  # False positive
                elif row['koi_disposition'] == 'CANDIDATE':
                    # Use probabilistic disposition if available
                    if pd.notna(row.get('koi_pdisposition')):
                        if row['koi_pdisposition'] == 'CANDIDATE':
                            labels[target_id] = 1  # Treat as exoplanet
                        else:
                            labels[target_id] = 0  # Treat as false positive
                    else:
                        labels[target_id] = 1  # Default to exoplanet for candidates
        
        elif mission.lower() == 'tess':
            for _, row in catalog.iterrows():
                target_id = f"TIC {int(row['tid'])}"
                
                if row['tfopwg_disp'] in ['CP', 'PC', 'KP', 'APC']:  # Confirmed/Planet Candidate/Known/Awaiting
                    labels[target_id] = 1
                elif row['tfopwg_disp'] in ['FP', 'FA']:  # False Positive/False Alarm
                    labels[target_id] = 0
                else:
                    labels[target_id] = 1  # Default to exoplanet
        
        elif mission.lower() == 'k2':
            for _, row in catalog.iterrows():
                # Extract EPIC number from epic_hostname (format: "EPIC 210848071")
                epic_host = row['epic_hostname']
                if isinstance(epic_host, str) and epic_host.startswith('EPIC '):
                    target_id = epic_host
                else:
                    continue  # Skip if we can't extract EPIC ID
                
                if row['disposition'] == 'CONFIRMED':
                    labels[target_id] = 1
                elif row['disposition'] in ['FALSE POSITIVE', 'REFUTED']:
                    labels[target_id] = 0
                elif row['disposition'] == 'CANDIDATE':
                    labels[target_id] = 1  # Treat candidates as exoplanets
                else:
                    labels[target_id] = 1  # Default to exoplanet
        
        self._labels[mission] = labels
        
        n_exoplanets = sum(labels.values())
        n_false_positives = len(labels) - n_exoplanets
        
        logger.info(f"{mission} labels: {n_exoplanets} exoplanets, {n_false_positives} false positives")
        
        return labels
    
    def get_stellar_features(self, mission: str) -> pd.DataFrame:
        """Extract stellar features from catalog.
        
        Args:
            mission: Mission name
            
        Returns:
            DataFrame with stellar features
        """
        catalog = self.load_catalog(mission)
        
        if mission.lower() == 'kepler':
            feature_cols = {
                'kepid': 'target_id',
                'koi_srad': 'stellar_radius',
                'koi_smass': 'stellar_mass',
                'koi_sage': 'stellar_age',
                'koi_smet': 'stellar_metallicity',
                'koi_kepmag': 'magnitude'
            }
            
            features = catalog[list(feature_cols.keys())].copy()
            features = features.rename(columns=feature_cols)
            features['target_id'] = features['target_id'].apply(lambda x: f"KIC {int(x)}")
        
        elif mission.lower() == 'tess':
            feature_cols = {
                'tid': 'target_id',
                'toi_srad': 'stellar_radius',
                'toi_smass': 'stellar_mass',
                'toi_sage': 'stellar_age',
                'toi_smet': 'stellar_metallicity',
                'toi_tmag': 'magnitude'
            }
            
            features = catalog[list(feature_cols.keys())].copy()
            features = features.rename(columns=feature_cols)
            features['target_id'] = features['target_id'].apply(lambda x: f"TIC {int(x)}")
        
        elif mission.lower() == 'k2':
            feature_cols = {
                'epic_number': 'target_id',
                'k2c_srad': 'stellar_radius',
                'k2c_smass': 'stellar_mass',
                'k2c_sage': 'stellar_age',
                'k2c_smet': 'stellar_metallicity',
                'k2c_kepmag': 'magnitude'
            }
            
            features = catalog[list(feature_cols.keys())].copy()
            features = features.rename(columns=feature_cols)
            features['target_id'] = features['target_id'].apply(lambda x: f"EPIC {int(x)}")
        
        # Clean and fill missing values
        numeric_cols = ['stellar_radius', 'stellar_mass', 'stellar_age', 'stellar_metallicity', 'magnitude']
        
        for col in numeric_cols:
            if col in features.columns:
                # Convert to numeric and fill NaN with median
                features[col] = pd.to_numeric(features[col], errors='coerce')
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)
        
        features['mission'] = mission.lower()
        
        return features
    
    def get_transit_features(self, mission: str) -> pd.DataFrame:
        """Extract transit features from catalog.
        
        Args:
            mission: Mission name
            
        Returns:
            DataFrame with transit features
        """
        catalog = self.load_catalog(mission)
        
        if mission.lower() == 'kepler':
            feature_cols = {
                'kepid': 'target_id',
                'koi_period': 'period',
                'koi_duration': 'duration',
                'koi_depth': 'depth',
                'koi_impact': 'impact',
                'koi_prad': 'planet_radius',
                'koi_sma': 'semi_major_axis',
                'koi_eccen': 'eccentricity',
                'koi_incl': 'inclination',
                'koi_teq': 'equilibrium_temp'
            }
            
            features = catalog[list(feature_cols.keys())].copy()
            features = features.rename(columns=feature_cols)
            features['target_id'] = features['target_id'].apply(lambda x: f"KIC {int(x)}")
        
        elif mission.lower() == 'tess':
            feature_cols = {
                'tid': 'target_id',
                'toi_period': 'period',
                'toi_duration': 'duration',
                'toi_depth': 'depth',
                'toi_prad': 'planet_radius',
                'toi_sma': 'semi_major_axis',
                'toi_eccen': 'eccentricity',
                'toi_incl': 'inclination',
                'toi_teq': 'equilibrium_temp'
            }
            
            features = catalog[list(feature_cols.keys())].copy()
            features = features.rename(columns=feature_cols)
            features['target_id'] = features['target_id'].apply(lambda x: f"TIC {int(x)}")
        
        elif mission.lower() == 'k2':
            feature_cols = {
                'epic_number': 'target_id',
                'k2c_period': 'period',
                'k2c_duration': 'duration',
                'k2c_depth': 'depth',
                'k2c_impact': 'impact',
                'k2c_prad': 'planet_radius',
                'k2c_sma': 'semi_major_axis',
                'k2c_eccen': 'eccentricity',
                'k2c_incl': 'inclination',
                'k2c_teq': 'equilibrium_temp'
            }
            
            features = catalog[list(feature_cols.keys())].copy()
            features = features.rename(columns=feature_cols)
            features['target_id'] = features['target_id'].apply(lambda x: f"EPIC {int(x)}")
        
        # Clean numeric columns
        numeric_cols = ['period', 'duration', 'depth', 'impact', 'planet_radius', 
                       'semi_major_axis', 'eccentricity', 'inclination', 'equilibrium_temp']
        
        for col in numeric_cols:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                # Fill NaN with median for each column
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)
        
        features['mission'] = mission.lower()
        
        return features
    
    def create_train_test_split(
        self, 
        target_ids: List[str], 
        labels: Dict[str, int],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """Create train/validation/test splits.
        
        Args:
            target_ids: List of target identifiers
            labels: Dictionary mapping target_id to label
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            stratify: Whether to stratify by labels
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids, train_labels, val_labels, test_labels)
        """
        # Filter target_ids to only include those with labels
        valid_ids = [tid for tid in target_ids if tid in labels]
        valid_labels = [labels[tid] for tid in valid_ids]
        
        logger.info(f"Creating splits from {len(valid_ids)} valid targets")
        
        if len(valid_ids) == 0:
            raise ValueError("No valid target IDs with labels found")
        
        # First split: train+val vs test
        train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
            valid_ids, valid_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=valid_labels if stratify else None
        )
        
        # Second split: train vs val
        if val_size > 0:
            # Adjust val_size relative to train+val set
            adjusted_val_size = val_size / (1 - test_size)
            
            train_ids, val_ids, train_labels, val_labels = train_test_split(
                train_val_ids, train_val_labels,
                test_size=adjusted_val_size,
                random_state=random_state,
                stratify=train_val_labels if stratify else None
            )
        else:
            train_ids, val_ids = train_val_ids, []
            train_labels, val_labels = train_val_labels, []
        
        logger.info(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        
        # Log class distribution
        for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
            if split_labels:
                n_positive = sum(split_labels)
                n_negative = len(split_labels) - n_positive
                logger.info(f"{split_name} - Positive: {n_positive}, Negative: {n_negative}")
        
        return train_ids, val_ids, test_ids, train_labels, val_labels, test_labels
    
    def get_combined_features(self, missions: List[str]) -> pd.DataFrame:
        """Get combined stellar and transit features for multiple missions.
        
        Args:
            missions: List of mission names
            
        Returns:
            Combined features DataFrame
        """
        all_features = []
        
        for mission in missions:
            try:
                stellar = self.get_stellar_features(mission)
                transit = self.get_transit_features(mission)
                
                # Merge on target_id
                features = pd.merge(stellar, transit, on=['target_id', 'mission'], how='inner')
                all_features.append(features)
                
                logger.info(f"Added {len(features)} features from {mission}")
                
            except Exception as e:
                logger.warning(f"Could not load features for {mission}: {e}")
        
        if not all_features:
            raise ValueError("No features loaded from any mission")
        
        combined = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined features: {len(combined)} total entries")
        
        return combined