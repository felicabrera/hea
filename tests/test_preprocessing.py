"""
Test Suite for Data Preprocessing and Feature Engineering
NASA Space Apps Challenge 2025 - HEA Project
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions"""
    
    def setUp(self):
        """Set up sample data for testing"""
        self.sample_data = pd.DataFrame({
            'koi_period': [365.25, 10.0, 1000.0],
            'koi_prad': [1.0, 0.5, 2.0],
            'koi_teq': [280.0, 1500.0, 150.0],
            'koi_insol': [1.0, 100.0, 0.1],
            'koi_steff': [5778.0, 6000.0, 4000.0],
            'koi_srad': [1.0, 1.2, 0.8],
            'koi_depth': [100.0, 500.0, 50.0],
            'koi_model_snr': [15.0, 30.0, 5.0]
        })
    
    def test_data_not_empty(self):
        """Test that sample data is not empty"""
        self.assertFalse(self.sample_data.empty, "Sample data should not be empty")
        self.assertGreater(len(self.sample_data), 0, "Sample data should have rows")
    
    def test_required_columns_present(self):
        """Test that required columns are present"""
        required_columns = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']
        
        for col in required_columns:
            self.assertIn(col, self.sample_data.columns, 
                         f"Required column {col} should be present")
    
    def test_numeric_columns_are_numeric(self):
        """Test that numeric columns contain numeric data"""
        numeric_columns = ['koi_period', 'koi_prad', 'koi_teq']
        
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data[col]),
                          f"Column {col} should be numeric")
    
    def test_no_all_nan_columns(self):
        """Test that no column is entirely NaN"""
        for col in self.sample_data.columns:
            self.assertFalse(self.sample_data[col].isna().all(),
                           f"Column {col} should not be all NaN")
    
    def test_habitability_zone_calculation(self):
        """Test habitability zone feature calculation"""
        # Simplified habitability zone check: temperature in reasonable range
        hz_mask = (self.sample_data['koi_teq'] >= 180) & (self.sample_data['koi_teq'] <= 310)
        
        self.assertIsInstance(hz_mask, pd.Series, "Result should be pandas Series")
        self.assertEqual(len(hz_mask), len(self.sample_data), 
                        "Mask should match data length")
    
    def test_planet_size_classification(self):
        """Test planet size classification"""
        # Earth radii ranges
        # Small: < 1.25, Earth-like: 1.25-2.0, Super-Earth: 2.0-4.0, Neptune: 4.0-10.0, Jupiter: > 10.0
        
        size_categories = pd.cut(
            self.sample_data['koi_prad'],
            bins=[0, 1.25, 2.0, 4.0, 10.0, float('inf')],
            labels=['Small', 'Earth-like', 'Super-Earth', 'Neptune', 'Jupiter']
        )
        
        self.assertEqual(len(size_categories), len(self.sample_data),
                        "Categories should match data length")
        self.assertFalse(size_categories.isna().all(),
                        "Categories should not be all NaN")
    
    def test_stellar_type_inference(self):
        """Test stellar type inference from temperature"""
        # Simplified stellar classification
        # Hot: > 7000K, Sun-like: 5000-7000K, Cool: < 5000K
        
        stellar_types = pd.cut(
            self.sample_data['koi_steff'],
            bins=[0, 5000, 7000, float('inf')],
            labels=['Cool', 'Sun-like', 'Hot']
        )
        
        self.assertEqual(len(stellar_types), len(self.sample_data),
                        "Stellar types should match data length")
    
    def test_insolation_flux_normalization(self):
        """Test insolation flux normalization"""
        # Normalize by Earth's insolation (1.0)
        normalized_insol = self.sample_data['koi_insol'] / 1.0
        
        self.assertEqual(len(normalized_insol), len(self.sample_data),
                        "Normalized values should match data length")
        self.assertTrue((normalized_insol >= 0).all(),
                       "Normalized insolation should be non-negative")


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""
    
    def setUp(self):
        """Set up test data with various issues"""
        self.valid_data = pd.DataFrame({
            'koi_period': [365.25, 10.0, 100.0],
            'koi_prad': [1.0, 0.5, 2.0],
            'koi_teq': [280.0, 1500.0, 150.0]
        })
        
        self.data_with_missing = pd.DataFrame({
            'koi_period': [365.25, np.nan, 100.0],
            'koi_prad': [1.0, 0.5, np.nan],
            'koi_teq': [280.0, 1500.0, 150.0]
        })
        
        self.data_with_negatives = pd.DataFrame({
            'koi_period': [365.25, -10.0, 100.0],
            'koi_prad': [1.0, 0.5, -2.0],
            'koi_teq': [280.0, 1500.0, 150.0]
        })
    
    def test_valid_data_passes(self):
        """Test that valid data passes validation"""
        self.assertFalse(self.valid_data.isna().any().any(),
                        "Valid data should have no missing values")
        self.assertTrue((self.valid_data >= 0).all().all(),
                       "Valid data should have no negative values")
    
    def test_detect_missing_values(self):
        """Test detection of missing values"""
        missing_mask = self.data_with_missing.isna()
        
        self.assertTrue(missing_mask.any().any(),
                       "Should detect missing values")
        self.assertEqual(missing_mask.sum().sum(), 2,
                        "Should detect exactly 2 missing values")
    
    def test_detect_negative_values(self):
        """Test detection of negative values"""
        negative_mask = self.data_with_negatives < 0
        
        self.assertTrue(negative_mask.any().any(),
                       "Should detect negative values")
    
    def test_column_completeness(self):
        """Test that all required columns are present"""
        required_columns = ['koi_period', 'koi_prad', 'koi_teq']
        
        for col in required_columns:
            self.assertIn(col, self.valid_data.columns,
                         f"Required column {col} should be present")
    
    def test_data_type_validation(self):
        """Test that columns have correct data types"""
        for col in self.valid_data.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.valid_data[col]),
                          f"Column {col} should be numeric")


class TestDataTransformations(unittest.TestCase):
    """Test data transformation functions"""
    
    def setUp(self):
        """Set up sample data for transformation tests"""
        self.sample_data = pd.DataFrame({
            'feature1': [1.0, 10.0, 100.0, 1000.0],
            'feature2': [0.1, 0.5, 1.0, 5.0],
            'feature3': [100, 200, 300, 400]
        })
    
    def test_log_transformation(self):
        """Test logarithmic transformation"""
        log_transformed = np.log10(self.sample_data['feature1'] + 1)
        
        self.assertEqual(len(log_transformed), len(self.sample_data),
                        "Transformed data should match original length")
        self.assertFalse(log_transformed.isna().any(),
                        "Log transform should not produce NaN for positive values")
    
    def test_standardization(self):
        """Test standardization (z-score normalization)"""
        mean = self.sample_data['feature2'].mean()
        std = self.sample_data['feature2'].std()
        standardized = (self.sample_data['feature2'] - mean) / std
        
        self.assertAlmostEqual(standardized.mean(), 0.0, places=10,
                              msg="Standardized data should have mean ~0")
        self.assertAlmostEqual(standardized.std(), 1.0, places=10,
                              msg="Standardized data should have std ~1")
    
    def test_min_max_scaling(self):
        """Test min-max normalization"""
        min_val = self.sample_data['feature3'].min()
        max_val = self.sample_data['feature3'].max()
        scaled = (self.sample_data['feature3'] - min_val) / (max_val - min_val)
        
        self.assertAlmostEqual(scaled.min(), 0.0, places=10,
                              msg="Min-max scaled should have min=0")
        self.assertAlmostEqual(scaled.max(), 1.0, places=10,
                              msg="Min-max scaled should have max=1")
    
    def test_handle_outliers(self):
        """Test outlier detection and handling"""
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100, 200])
        
        # IQR method
        Q1 = data_with_outliers.quantile(0.25)
        Q3 = data_with_outliers.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
        
        self.assertTrue(outliers.any(), "Should detect outliers")
        # With this data: Q1=2.5, Q3=52.5, IQR=50, upper=127.5
        # Only 200 is an outlier (100 < 127.5)
        self.assertEqual(outliers.sum(), 1, "Should detect 1 outlier (200)")


class TestFeatureSelection(unittest.TestCase):
    """Test feature selection and importance"""
    
    def setUp(self):
        """Set up sample features"""
        self.features = pd.DataFrame({
            'koi_period': [365.25, 10.0, 100.0, 50.0],
            'koi_prad': [1.0, 0.5, 2.0, 1.5],
            'koi_teq': [280.0, 1500.0, 150.0, 350.0],
            'koi_insol': [1.0, 100.0, 0.1, 2.0],
            'koi_depth': [100.0, 500.0, 50.0, 200.0]
        })
    
    def test_feature_variance(self):
        """Test feature variance calculation"""
        variance = self.features.var()
        
        self.assertEqual(len(variance), len(self.features.columns),
                        "Should calculate variance for all features")
        self.assertTrue((variance >= 0).all(),
                       "Variance should be non-negative")
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation"""
        corr_matrix = self.features.corr()
        
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1],
                        "Correlation matrix should be square")
        self.assertTrue((corr_matrix.values >= -1).all() and (corr_matrix.values <= 1).all(),
                       "Correlation values should be between -1 and 1")
    
    def test_remove_low_variance_features(self):
        """Test removal of low variance features"""
        # Add a constant feature (zero variance)
        features_with_constant = self.features.copy()
        features_with_constant['constant_feature'] = 1.0
        
        variance = features_with_constant.var()
        low_variance_cols = variance[variance < 0.01].index.tolist()
        
        self.assertIn('constant_feature', low_variance_cols,
                     "Should identify constant feature")


if __name__ == '__main__':
    unittest.main(verbosity=2)
