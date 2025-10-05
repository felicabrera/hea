"""
Test Suite for Data Loading and Processing
NASA Space Apps Challenge 2025 - HEA Project
"""

import unittest
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataFiles(unittest.TestCase):
    """Test data file existence and structure"""
    
    @classmethod
    def setUpClass(cls):
        """Set up paths for data tests"""
        cls.data_dir = Path(__file__).parent.parent / "data"
        cls.catalogs_dir = cls.data_dir / "catalogs"
    
    def test_data_directory_exists(self):
        """Test that data directory exists"""
        self.assertTrue(self.data_dir.exists(), "Data directory should exist")
    
    def test_catalogs_directory_exists(self):
        """Test that catalogs directory exists"""
        self.assertTrue(self.catalogs_dir.exists(), "Catalogs directory should exist")
    
    def test_kepler_catalog_exists(self):
        """Test that Kepler catalog file exists"""
        kepler_file = self.catalogs_dir / "kepler_koi.csv"
        self.assertTrue(kepler_file.exists(), "Kepler KOI catalog should exist")
    
    def test_tess_catalog_exists(self):
        """Test that TESS catalog file exists"""
        tess_file = self.catalogs_dir / "tess_toi.csv"
        self.assertTrue(tess_file.exists(), "TESS TOI catalog should exist")
    
    def test_k2_catalog_exists(self):
        """Test that K2 catalog file exists"""
        k2_file = self.catalogs_dir / "k2_catalog.csv"
        self.assertTrue(k2_file.exists(), "K2 catalog should exist")


class TestCatalogStructure(unittest.TestCase):
    """Test catalog file structure and content"""
    
    @classmethod
    def setUpClass(cls):
        """Load catalog files for testing"""
        cls.catalogs_dir = Path(__file__).parent.parent / "data" / "catalogs"
        cls.kepler_file = cls.catalogs_dir / "kepler_koi.csv"
        cls.tess_file = cls.catalogs_dir / "tess_toi.csv"
        cls.k2_file = cls.catalogs_dir / "k2_catalog.csv"
    
    def test_kepler_catalog_loads(self):
        """Test that Kepler catalog can be loaded"""
        if self.kepler_file.exists():
            try:
                df = pd.read_csv(self.kepler_file, comment='#')
                self.assertIsInstance(df, pd.DataFrame, "Should load as DataFrame")
                self.assertGreater(len(df), 0, "Should contain data")
            except Exception as e:
                self.fail(f"Failed to load Kepler catalog: {e}")
    
    def test_tess_catalog_loads(self):
        """Test that TESS catalog can be loaded"""
        if self.tess_file.exists():
            try:
                df = pd.read_csv(self.tess_file, comment='#')
                self.assertIsInstance(df, pd.DataFrame, "Should load as DataFrame")
                self.assertGreater(len(df), 0, "Should contain data")
            except Exception as e:
                self.fail(f"Failed to load TESS catalog: {e}")
    
    def test_k2_catalog_loads(self):
        """Test that K2 catalog can be loaded"""
        if self.k2_file.exists():
            try:
                df = pd.read_csv(self.k2_file, comment='#')
                self.assertIsInstance(df, pd.DataFrame, "Should load as DataFrame")
                self.assertGreater(len(df), 0, "Should contain data")
            except Exception as e:
                self.fail(f"Failed to load K2 catalog: {e}")
    
    def test_kepler_required_columns(self):
        """Test that Kepler catalog has required columns"""
        if self.kepler_file.exists():
            try:
                df = pd.read_csv(self.kepler_file, comment='#')
                required_columns = ['koi_period', 'koi_disposition']
                
                for col in required_columns:
                    self.assertIn(col, df.columns, f"Kepler catalog should have {col} column")
            except Exception as e:
                self.fail(f"Column check failed: {e}")
    
    def test_k2_required_columns(self):
        """Test that K2 catalog has required columns"""
        if self.k2_file.exists():
            try:
                df = pd.read_csv(self.k2_file, comment='#')
                required_columns = ['pl_name', 'disposition']
                
                for col in required_columns:
                    self.assertIn(col, df.columns, f"K2 catalog should have {col} column")
            except Exception as e:
                self.fail(f"Column check failed: {e}")
    
    def test_tess_required_columns(self):
        """Test that TESS catalog has required columns"""
        if self.tess_file.exists():
            try:
                df = pd.read_csv(self.tess_file, comment='#')
                required_columns = ['toi', 'tfopwg_disp']
                
                for col in required_columns:
                    self.assertIn(col, df.columns, f"TESS catalog should have {col} column")
            except Exception as e:
                self.fail(f"Column check failed: {e}")
    
    def test_catalog_data_quality(self):
        """Test basic data quality of catalogs"""
        for catalog_file in [self.kepler_file, self.tess_file, self.k2_file]:
            if catalog_file.exists():
                try:
                    df = pd.read_csv(catalog_file, comment='#')
                    
                    # Check not all NaN
                    self.assertFalse(df.isna().all().all(), 
                                   f"{catalog_file.name} should not be all NaN")
                    
                    # Check has rows
                    self.assertGreater(len(df), 0, 
                                     f"{catalog_file.name} should have rows")
                    
                    # Check has columns
                    self.assertGreater(len(df.columns), 0, 
                                     f"{catalog_file.name} should have columns")
                except Exception as e:
                    self.fail(f"Data quality check failed for {catalog_file.name}: {e}")


class TestHabitabilityRankings(unittest.TestCase):
    """Test habitability rankings file"""
    
    @classmethod
    def setUpClass(cls):
        """Set up paths for rankings tests"""
        cls.rankings_file = Path(__file__).parent.parent / "top_habitable_candidates.csv"
    
    def test_rankings_file_exists_or_can_be_generated(self):
        """Test that rankings file exists or can be generated"""
        # File may not exist initially, which is acceptable
        if self.rankings_file.exists():
            self.assertTrue(True, "Rankings file exists")
        else:
            # Check if script exists to generate it
            script_file = Path(__file__).parent.parent / "scripts" / "analyze_habitability.py"
            self.assertTrue(script_file.exists(), 
                          "Habitability analysis script should exist")
    
    def test_rankings_structure_if_exists(self):
        """Test rankings file structure if it exists"""
        if self.rankings_file.exists():
            try:
                df = pd.read_csv(self.rankings_file)
                
                # Check required columns
                required_columns = [
                    'candidate_name', 
                    'mission', 
                    'hab_habitability_score',
                    'hab_habitability_class'
                ]
                
                for col in required_columns:
                    self.assertIn(col, df.columns, f"Rankings should have {col} column")
                
                # Check habitability scores are valid
                if 'hab_habitability_score' in df.columns:
                    scores = df['hab_habitability_score'].dropna()
                    if len(scores) > 0:
                        self.assertTrue(scores.between(0, 1).all() or scores.between(0, 100).all(),
                                      "Habitability scores should be in valid range")
            except Exception as e:
                self.fail(f"Rankings structure test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
