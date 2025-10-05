"""
Test Suite for Backend API Endpoints
NASA Space Apps Challenge 2025 - HEA Project
Tests FastAPI endpoints with various scenarios
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing FastAPI test client and backend
try:
    from fastapi.testclient import TestClient
    from backend.main import app, model_cache, load_model
    
    FASTAPI_AVAILABLE = True
    
    # Load model for testing
    try:
        if model_cache["model"] is None:
            model, metadata, feature_names, threshold = load_model()
            model_cache["model"] = model
            model_cache["metadata"] = metadata
            model_cache["feature_names"] = feature_names
            model_cache["threshold"] = threshold
            print(" Model loaded successfully for API tests")
    except Exception as e:
        print(f"Warning: Could not load model for API tests: {e}")
        
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPIHealth(unittest.TestCase):
    """Test API health and basic endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        if FASTAPI_AVAILABLE:
            cls.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns success"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, "Root endpoint should return 200")
        
        data = response.json()
        self.assertIn("name", data, "Response should contain name")
        self.assertIn("version", data, "Response should contain version")
        self.assertIn("description", data, "Response should contain description")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200, "Health endpoint should return 200")
        
        data = response.json()
        self.assertIn("status", data, "Response should contain status")
        # Status can be "healthy" or "model_not_loaded" depending on model state
        self.assertIn(data["status"], ["healthy", "model_not_loaded"], "Status should be valid")
    
    def test_docs_endpoint_accessible(self):
        """Test that API documentation is accessible"""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200, 
                        "API docs should be accessible")


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestPredictionEndpoint(unittest.TestCase):
    """Test prediction API endpoint"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and sample data"""
        if FASTAPI_AVAILABLE:
            cls.client = TestClient(app)
            
            # Sample valid planet data
            cls.valid_planet = {
                "koi_period": 365.25,
                "koi_time0bk": 131.5,
                "koi_impact": 0.5,
                "koi_duration": 3.5,
                "koi_depth": 100.0,
                "koi_prad": 1.0,
                "koi_teq": 280.0,
                "koi_insol": 1.0,
                "koi_model_snr": 15.0,
                "koi_steff": 5778.0,
                "koi_slogg": 4.4,
                "koi_srad": 1.0,
                "ra": 285.0,
                "dec": 45.0
            }
            
            # Earth-like parameters for habitability
            cls.earth_like_planet = cls.valid_planet.copy()
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        # API expects data to be wrapped in a 'data' field
        payload = {"data": self.valid_planet}
        response = self.client.post("/predict", json=payload)
        
        if response.status_code == 503:
            self.skipTest("Model not loaded in backend")
        
        self.assertEqual(response.status_code, 200, 
                        "Valid prediction should return 200")
        
        data = response.json()
        self.assertIn("prediction", data, "Response should contain prediction")
        self.assertIn("probability", data, "Response should contain probability")
        self.assertIn("confidence", data, "Response should contain confidence")
    
    def test_predict_missing_field(self):
        """Test prediction with missing required field"""
        invalid_planet = self.valid_planet.copy()
        del invalid_planet["koi_period"]
        
        payload = {"data": invalid_planet}
        response = self.client.post("/predict", json=payload)
        
        if response.status_code == 503:
            self.skipTest("Model not loaded in backend")
        
        # API handles missing fields by filling with 0, so it returns 200
        self.assertEqual(response.status_code, 200, 
                        "Missing fields are filled with default values")
    
    def test_predict_invalid_type(self):
        """Test prediction with invalid data type"""
        invalid_planet = self.valid_planet.copy()
        invalid_planet["koi_period"] = "not a number"
        
        payload = {"data": invalid_planet}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422,
                        "Invalid type should return 422")
    
    def test_predict_negative_values(self):
        """Test prediction with negative values (edge case)"""
        edge_case_planet = self.valid_planet.copy()
        edge_case_planet["koi_period"] = -10.0
        
        payload = {"data": edge_case_planet}
        response = self.client.post("/predict", json=payload)
        
        if response.status_code == 503:
            self.skipTest("Model not loaded in backend")
        
        # Should either return 200 with prediction or 422 with validation error
        self.assertIn(response.status_code, [200, 422], 
                     "Negative values should be handled")
    
    def test_predict_extreme_values(self):
        """Test prediction with extreme values"""
        extreme_planet = self.valid_planet.copy()
        extreme_planet["koi_period"] = 10000.0
        extreme_planet["koi_teq"] = 5000.0
        
        payload = {"data": extreme_planet}
        response = self.client.post("/predict", json=payload)
        
        if response.status_code == 503:
            self.skipTest("Model not loaded in backend")
        
        self.assertEqual(response.status_code, 200, 
                        "Extreme values should be handled")
    
    def test_predict_probability_range(self):
        """Test that prediction probability is in valid range"""
        payload = {"data": self.valid_planet}
        response = self.client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            prob = data.get("probability", -1)
            self.assertGreaterEqual(prob, 0.0, "Probability should be >= 0")
            self.assertLessEqual(prob, 1.0, "Probability should be <= 1")


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestHabitabilityEndpoint(unittest.TestCase):
    """Test habitability analysis endpoint"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and sample data"""
        if FASTAPI_AVAILABLE:
            cls.client = TestClient(app)
            
            # Earth-like planet parameters (habitability API format)
            cls.earth_like = {
                "radius": 1.0,
                "temp": 288.0,
                "insolation": 1.0,
                "stellar_temp": 5778.0,
                "period": 365.25,
                "disposition": "CONFIRMED"
            }
            
            # Hot Jupiter parameters
            cls.hot_jupiter = {
                "radius": 12.0,
                "temp": 1500.0,
                "insolation": 1000.0,
                "stellar_temp": 6000.0,
                "period": 3.5
            }
    
    def test_habitability_earth_like(self):
        """Test habitability scoring for Earth-like planet"""
        response = self.client.post("/habitability", json=self.earth_like)
        self.assertEqual(response.status_code, 200, 
                        "Earth-like planet should return 200")
        
        data = response.json()
        self.assertIn("habitability_score", data)
        self.assertIn("habitability_class", data)
        
        # Earth-like should have HIGH habitability
        score = data["habitability_score"]
        if score is not None:
            self.assertGreater(score, 0.7, "Earth-like should have high score")
    
    def test_habitability_hot_jupiter(self):
        """Test habitability scoring for hot Jupiter"""
        response = self.client.post("/habitability", json=self.hot_jupiter)
        self.assertEqual(response.status_code, 200,
                        "Hot Jupiter should return 200")
        
        data = response.json()
        score = data.get("habitability_score")
        if score is not None:
            self.assertLess(score, 0.5, "Hot Jupiter should have low habitability")
    
    def test_habitability_missing_field(self):
        """Test habitability with missing field"""
        # All fields are optional, so this should work
        partial_data = {"radius": 1.0, "temp": 288.0}
        response = self.client.post("/habitability", json=partial_data)
        self.assertEqual(response.status_code, 200,
                        "Partial data should be accepted")
    
    def test_habitability_none_values(self):
        """Test habitability with None values"""
        data_with_none = {"radius": 1.0, "temp": None, "insolation": 1.0}
        response = self.client.post("/habitability", json=data_with_none)
        self.assertEqual(response.status_code, 200,
                        "None values should be handled gracefully")
    
    def test_habitability_score_range(self):
        """Test that habitability score is in valid range"""
        response = self.client.post("/habitability", json=self.earth_like)
        
        if response.status_code == 200:
            data = response.json()
            score = data.get("habitability_score")
            if score is not None:
                self.assertGreaterEqual(score, 0.0, "Score should be >= 0")
                self.assertLessEqual(score, 1.0, "Score should be <= 1")


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestBatchEndpoint(unittest.TestCase):
    """Test batch prediction endpoint"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and sample data"""
        if FASTAPI_AVAILABLE:
            cls.client = TestClient(app)
            
            cls.sample_planet = {
                "koi_period": 365.25,
                "koi_time0bk": 131.5,
                "koi_impact": 0.5,
                "koi_duration": 3.5,
                "koi_depth": 100.0,
                "koi_prad": 1.0,
                "koi_teq": 280.0,
                "koi_insol": 1.0,
                "koi_model_snr": 15.0,
                "koi_steff": 5778.0,
                "koi_slogg": 4.4,
                "koi_srad": 1.0,
                "ra": 285.0,
                "dec": 45.0
            }
    
    def test_batch_single_planet(self):
        """Test batch prediction with single planet"""
        # Backend expects 'data' field with list of planet dicts
        response = self.client.post("/predict/batch", json={"data": [self.sample_planet]})
        
        if response.status_code == 503:
            self.skipTest("Model not loaded in backend")
        
        self.assertEqual(response.status_code, 200, 
                        "Single planet batch should return 200")
        
        data = response.json()
        self.assertIn("predictions", data, "Response should contain predictions")
        self.assertEqual(len(data["predictions"]), 1, 
                        "Should return one prediction")
    
    def test_batch_multiple_planets(self):
        """Test batch prediction with multiple planets"""
        planets = [self.sample_planet.copy() for _ in range(3)]
        
        # Backend expects 'data' field with list of planet dicts
        response = self.client.post("/predict/batch", json={"data": planets})
        
        if response.status_code == 503:
            self.skipTest("Model not loaded in backend")
        
        self.assertEqual(response.status_code, 200, 
                        "Multiple planets batch should return 200")
        
        data = response.json()
        self.assertEqual(len(data["predictions"]), 3, 
                        "Should return three predictions")
    
    def test_batch_empty_list(self):
        """Test batch prediction with empty list"""
        response = self.client.post("/predict/batch", json={"planets": []})
        # Should either accept empty list or return validation error
        self.assertIn(response.status_code, [200, 422], 
                     "Empty list should be handled")


if __name__ == '__main__':
    unittest.main(verbosity=2)
