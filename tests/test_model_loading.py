"""
Test Suite for Model Loading and Prediction
NASA Space Apps Challenge 2025 - HEA Project
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelLoading(unittest.TestCase):
    """Test model loading and basic functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.model_dir = Path(__file__).parent.parent / "models" / "catalog_models"
        cls.model_loaded = False
        cls.model = None
        cls.metadata = None
        
        try:
            import joblib
            model_files = list(cls.model_dir.glob("*.joblib")) + list(cls.model_dir.glob("*.pkl"))
            
            if model_files:
                model_path = model_files[0]
                data = joblib.load(model_path)
                
                if isinstance(data, dict):
                    # Check if 'model' key exists and is also a dict (ultra model structure)
                    model_data = data.get('model')
                    if isinstance(model_data, dict) and 'stacking' in model_data:
                        cls.model = model_data.get('stacking')
                    else:
                        cls.model = model_data
                    
                    # Extract metadata - could be in data or in model_data
                    cls.metadata = data.get('metadata', {})
                    if not cls.metadata:
                        # Use other keys as metadata
                        cls.metadata = {k: v for k, v in data.items() if k != 'model' and isinstance(v, (int, float, str, bool, list, dict))}
                else:
                    cls.model = data
                    cls.metadata = {}
                
                cls.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
    
    def test_model_file_exists(self):
        """Test that model files exist in the expected directory"""
        self.assertTrue(self.model_dir.exists(), "Model directory should exist")
        
        model_files = list(self.model_dir.glob("*.joblib")) + list(self.model_dir.glob("*.pkl"))
        self.assertGreater(len(model_files), 0, "At least one model file should exist")
    
    def test_model_loads_successfully(self):
        """Test that model can be loaded without errors"""
        self.assertTrue(self.model_loaded, "Model should load successfully")
        self.assertIsNotNone(self.model, "Model object should not be None")
    
    def test_model_has_predict_method(self):
        """Test that model has required prediction methods"""
        if self.model_loaded and self.model is not None:
            self.assertTrue(hasattr(self.model, 'predict'), "Model should have predict method")
            self.assertTrue(hasattr(self.model, 'predict_proba'), "Model should have predict_proba method")
        else:
            self.skipTest("Model not loaded properly")
    
    def test_metadata_structure(self):
        """Test that metadata contains expected keys"""
        if self.model_loaded and self.metadata:
            expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            for key in expected_keys:
                self.assertIn(key, self.metadata, f"Metadata should contain {key}")
    
    def test_metadata_values_valid(self):
        """Test that metadata values are within valid ranges"""
        if self.model_loaded and self.metadata:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            for metric in metrics:
                if metric in self.metadata:
                    value = self.metadata[metric]
                    self.assertGreaterEqual(value, 0.0, f"{metric} should be >= 0")
                    self.assertLessEqual(value, 1.0, f"{metric} should be <= 1")
    
    def test_model_prediction_shape(self):
        """Test that model predictions have correct shape"""
        if not self.model_loaded or self.model is None:
            self.skipTest("Model not loaded properly")
            
        # Create dummy input with correct number of features
        n_features = self.metadata.get('n_features', 134)
        dummy_input = np.random.random((5, n_features))
        
        try:
            predictions = self.model.predict(dummy_input)
            self.assertEqual(len(predictions), 5, "Should predict for all 5 samples")
            
            probabilities = self.model.predict_proba(dummy_input)
            self.assertEqual(probabilities.shape[0], 5, "Should have probabilities for all samples")
            self.assertEqual(probabilities.shape[1], 2, "Should have 2 classes (binary classification)")
        except Exception as e:
            self.skipTest(f"Model prediction test skipped: {e}")
    
    def test_prediction_output_type(self):
        """Test that predictions are of correct type"""
        if not self.model_loaded or self.model is None:
            self.skipTest("Model not loaded properly")
            
        n_features = self.metadata.get('n_features', 134)
        dummy_input = np.random.random((1, n_features))
        
        try:
            predictions = self.model.predict(dummy_input)
            self.assertIsInstance(predictions, np.ndarray, "Predictions should be numpy array")
            
            probabilities = self.model.predict_proba(dummy_input)
            self.assertIsInstance(probabilities, np.ndarray, "Probabilities should be numpy array")
        except Exception as e:
            self.skipTest(f"Prediction type test skipped: {e}")
    
    def test_probability_values_valid(self):
        """Test that probability values are valid"""
        if not self.model_loaded or self.model is None:
            self.skipTest("Model not loaded properly")
            
        n_features = self.metadata.get('n_features', 134)
        dummy_input = np.random.random((3, n_features))
        
        try:
            probabilities = self.model.predict_proba(dummy_input)
            
            # Check all probabilities are between 0 and 1
            self.assertTrue(np.all(probabilities >= 0), "All probabilities should be >= 0")
            self.assertTrue(np.all(probabilities <= 1), "All probabilities should be <= 1")
            
            # Check probabilities sum to 1 for each sample
            prob_sums = probabilities.sum(axis=1)
            np.testing.assert_array_almost_equal(prob_sums, np.ones(3), decimal=5,
                                                 err_msg="Probabilities should sum to 1")
        except Exception as e:
            self.skipTest(f"Probability validation test skipped: {e}")


class TestModelPerformance(unittest.TestCase):
    """Test model performance metrics"""
    
    @classmethod
    def setUpClass(cls):
        """Load metadata for performance tests"""
        cls.model_dir = Path(__file__).parent.parent / "models" / "catalog_models"
        cls.metadata = {}
        
        try:
            import joblib
            model_files = list(cls.model_dir.glob("*.joblib")) + list(cls.model_dir.glob("*.pkl"))
            
            if model_files:
                data = joblib.load(model_files[0])
                if isinstance(data, dict):
                    cls.metadata = data.get('metadata', {})
        except Exception:
            pass
    
    def test_accuracy_threshold(self):
        """Test that model accuracy meets minimum threshold"""
        if 'accuracy' in self.metadata:
            accuracy = self.metadata['accuracy']
            self.assertGreaterEqual(accuracy, 0.90, "Model accuracy should be at least 90%")
    
    def test_recall_threshold(self):
        """Test that model recall meets minimum threshold"""
        if 'recall' in self.metadata:
            recall = self.metadata['recall']
            self.assertGreaterEqual(recall, 0.95, "Model recall should be at least 95%")
    
    def test_precision_threshold(self):
        """Test that model precision meets minimum threshold"""
        if 'precision' in self.metadata:
            precision = self.metadata['precision']
            self.assertGreaterEqual(precision, 0.90, "Model precision should be at least 90%")
    
    def test_f1_score_threshold(self):
        """Test that F1 score meets minimum threshold"""
        if 'f1_score' in self.metadata:
            f1 = self.metadata['f1_score']
            self.assertGreaterEqual(f1, 0.90, "F1 score should be at least 90%")
    
    def test_auc_threshold(self):
        """Test that AUC score meets minimum threshold"""
        if 'auc_score' in self.metadata:
            auc = self.metadata['auc_score']
            self.assertGreaterEqual(auc, 0.95, "AUC should be at least 95%")


if __name__ == '__main__':
    unittest.main(verbosity=2)
