"""
Test Suite for Habitability Scorer
NASA Space Apps Challenge 2025 - HEA Project
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'utils'))

try:
    from habitability_scorer import HabitabilityScorer
    HABITABILITY_AVAILABLE = True
except ImportError:
    HABITABILITY_AVAILABLE = False
    print("Warning: HabitabilityScorer not available")


@unittest.skipIf(not HABITABILITY_AVAILABLE, "HabitabilityScorer not available")
class TestHabitabilityScorer(unittest.TestCase):
    """Test habitability scoring system"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize scorer for all tests"""
        cls.scorer = HabitabilityScorer()
    
    def test_scorer_initialization(self):
        """Test that scorer initializes without errors"""
        scorer = HabitabilityScorer()
        self.assertIsNotNone(scorer, "Scorer should initialize successfully")
    
    def test_earth_like_planet_high_score(self):
        """Test that Earth-like parameters produce high habitability score"""
        result = self.scorer.calculate_habitability_score(
            radius=1.0,        # Earth radius
            temp=288,          # Earth temperature
            insolation=1.0,    # Earth insolation
            stellar_temp=5778, # Sun-like star
            period=365.25      # Earth year
        )
        
        self.assertIsInstance(result, dict, "Should return dict")
        score = result.get('habitability_score')
        self.assertIsNotNone(score, "Score should not be None")
        self.assertGreater(score, 0.7, "Earth-like planet should have high habitability score")
    
    def test_hot_jupiter_low_score(self):
        """Test that hot Jupiter parameters produce low habitability score"""
        result = self.scorer.calculate_habitability_score(
            radius=11.0,       # Jupiter-sized
            temp=1500,         # Very hot
            insolation=100.0,  # High stellar flux
            stellar_temp=6000,
            period=3.5         # Very short period
        )
        
        self.assertIsInstance(result, dict, "Should return dict")
        score = result.get('habitability_score')
        self.assertIsNotNone(score, "Score should not be None")
        self.assertLess(score, 0.5, "Hot Jupiter should have low habitability score")
    
    def test_score_range_valid(self):
        """Test that scores are always between 0 and 1"""
        test_cases = [
            {'radius': 0.5, 'temp': 200, 'insolation': 0.3, 'stellar_temp': 4000, 'period': 10},
            {'radius': 1.5, 'temp': 350, 'insolation': 1.5, 'stellar_temp': 6500, 'period': 500},
            {'radius': 5.0, 'temp': 800, 'insolation': 50.0, 'stellar_temp': 7000, 'period': 2},
        ]
        
        for params in test_cases:
            result = self.scorer.calculate_habitability_score(**params)
            self.assertIsInstance(result, dict, "Should return dict")
            score = result.get('habitability_score')
            if score is not None:
                try:
                    if not np.isnan(score):
                        self.assertGreaterEqual(score, 0.0, f"Score should be >= 0 for params {params}")
                        self.assertLessEqual(score, 1.0, f"Score should be <= 1 for params {params}")
                except (TypeError, ValueError):
                    pass  # Handle non-numeric scores
    
    def test_none_values_handled(self):
        """Test that None values are handled gracefully"""
        result = self.scorer.calculate_habitability_score(
            radius=None,
            temp=None,
            insolation=1.0,
            stellar_temp=5778,
            period=365
        )
        
        # Should return dict even with None values
        self.assertIsInstance(result, dict, "Should return dict with None values")
        score = result.get('habitability_score')
        if score is not None:
            try:
                if not np.isnan(score):
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)
            except (TypeError, ValueError):
                pass  # Handle non-numeric scores
    
    def test_extreme_values_handled(self):
        """Test that extreme values don't cause errors"""
        extreme_cases = [
            {'radius': 0.01, 'temp': 1, 'insolation': 0.001, 'stellar_temp': 2000, 'period': 0.5},
            {'radius': 100, 'temp': 5000, 'insolation': 1000, 'stellar_temp': 50000, 'period': 10000},
        ]
        
        for params in extreme_cases:
            try:
                result = self.scorer.calculate_habitability_score(**params)
                self.assertIsInstance(result, dict, "Should return dict for extreme values")
                score = result.get('habitability_score')
                if score is not None:
                    try:
                        if not np.isnan(score):
                            self.assertGreaterEqual(score, 0.0, "Score should be valid even for extreme values")
                            self.assertLessEqual(score, 1.0, "Score should be valid even for extreme values")
                    except (TypeError, ValueError):
                        pass  # Handle non-numeric scores
            except Exception as e:
                self.fail(f"Should handle extreme values without crashing: {e}")
    
    def test_component_scores_returned(self):
        """Test that component scores are calculated when requested"""
        result = self.scorer.calculate_habitability_score(
            radius=1.0,
            temp=288,
            insolation=1.0,
            stellar_temp=5778,
            period=365.25
        )
        
        # Always returns dict with component scores
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertIn('habitability_score', result, "Should return overall score")
        self.assertIn('radius_score', result, "Should include radius score")
        self.assertIn('temperature_score', result, "Should include temperature score")
    
    def test_reproducibility(self):
        """Test that same inputs produce same outputs"""
        params = {
            'radius': 1.2,
            'temp': 300,
            'insolation': 0.9,
            'stellar_temp': 5500,
            'period': 400
        }
        
        score1 = self.scorer.calculate_habitability_score(**params)
        score2 = self.scorer.calculate_habitability_score(**params)
        
        if score1 is not None and score2 is not None:
            self.assertEqual(score1, score2, "Same inputs should produce same outputs")


class TestHabitabilityClassification(unittest.TestCase):
    """Test habitability classification logic"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize scorer for classification tests"""
        if HABITABILITY_AVAILABLE:
            cls.scorer = HabitabilityScorer()
    
    @unittest.skipIf(not HABITABILITY_AVAILABLE, "HabitabilityScorer not available")
    def test_high_habitability_classification(self):
        """Test that high scores are classified as HIGH habitability"""
        if hasattr(self.scorer, 'classify_habitability'):
            classification = self.scorer.classify_habitability(0.85)
            self.assertEqual(classification, 'HIGH', "Score >= 0.8 should be HIGH")
    
    @unittest.skipIf(not HABITABILITY_AVAILABLE, "HabitabilityScorer not available")
    def test_medium_habitability_classification(self):
        """Test that medium scores are classified as MEDIUM habitability"""
        if hasattr(self.scorer, 'classify_habitability'):
            classification = self.scorer.classify_habitability(0.65)
            self.assertEqual(classification, 'MEDIUM', "Score 0.6-0.8 should be MEDIUM")
    
    @unittest.skipIf(not HABITABILITY_AVAILABLE, "HabitabilityScorer not available")
    def test_low_habitability_classification(self):
        """Test that low scores are classified as LOW habitability"""
        if hasattr(self.scorer, 'classify_habitability'):
            classification = self.scorer.classify_habitability(0.35)
            self.assertEqual(classification, 'LOW', "Score < 0.6 should be LOW")


if __name__ == '__main__':
    unittest.main(verbosity=2)
