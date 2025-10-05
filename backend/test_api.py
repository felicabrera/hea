"""
Quick API Test Script
Tests all major endpoints of the HEA API
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def test_health():
    """Test health endpoint"""
    print(" Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Model Loaded: {data['model_loaded']}")
        print(f"   [OK] PASSED")
        return True
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n[DATA] Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        if response.status_code == 503:
            print(f"   WARNING:  Model not loaded")
            return False
        data = response.json()
        print(f"   Model: {data['model_name']}")
        print(f"   Accuracy: {data['accuracy']:.2%}")
        print(f"   Recall: {data['recall']:.2%}")
        print(f"   Precision: {data['precision']:.2%}")
        print(f"   Features: {data['n_features']}")
        print(f"   [OK] PASSED")
        return True
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
        return False

def test_single_prediction():
    """Test single prediction"""
    print("\n Testing single prediction endpoint...")
    try:
        payload = {
            "data": {
                "koi_period": 3.5225,
                "koi_depth": 200.5,
                "koi_duration": 2.5,
                "st_teff": 5778,
                "st_rad": 1.02,
                "koi_count": 45
            }
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        if response.status_code == 503:
            print(f"   WARNING:  Model not loaded")
            return False
        data = response.json()
        print(f"   Prediction: {data['prediction']}")
        print(f"   Probability: {data['probability']:.2%}")
        print(f"   Confidence: {data['confidence']}")
        print(f"   [OK] PASSED")
        return True
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
        return False

def test_prediction_with_explanation():
    """Test prediction with SHAP explanation"""
    print("\n Testing prediction with explanation endpoint...")
    try:
        payload = {
            "data": {
                "koi_period": 3.5225,
                "koi_depth": 200.5,
                "koi_duration": 2.5,
                "st_teff": 5778
            }
        }
        response = requests.post(f"{BASE_URL}/predict/explain?top_features=3", json=payload)
        if response.status_code == 503:
            print(f"   WARNING:  Model not loaded")
            return False
        data = response.json()
        print(f"   Prediction: {data['prediction']}")
        print(f"   Probability: {data['probability']:.2%}")
        
        if data.get('shap_explanation'):
            print(f"   Top Features:")
            for feat in data['shap_explanation']['top_features'][:3]:
                print(f"      - {feat['feature']}: {feat['direction']}")
        
        if data.get('base_model_votes'):
            print(f"   Base Model Votes:")
            for name, prob in data['base_model_votes'].items():
                print(f"      - {name.upper()}: {prob:.2%}")
        
        print(f"   [OK] PASSED")
        return True
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    print("\n Testing batch prediction endpoint...")
    try:
        payload = {
            "data": [
                {"koi_period": 3.5225, "koi_depth": 200.5, "st_teff": 5778},
                {"koi_period": 4.8921, "koi_depth": 180.3, "st_teff": 5912},
                {"koi_period": 7.3842, "koi_depth": 150.2, "st_teff": 6123}
            ]
        }
        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        if response.status_code == 503:
            print(f"   WARNING:  Model not loaded")
            return False
        data = response.json()
        print(f"   Total Processed: {data['total_processed']}")
        print(f"   Summary:")
        print(f"      - Exoplanets: {data['summary']['predicted_exoplanets']}")
        print(f"      - Non-Exoplanets: {data['summary']['predicted_non_exoplanets']}")
        print(f"      - Avg Probability: {data['summary']['avg_probability']:.2%}")
        print(f"   [OK] PASSED")
        return True
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
        return False

def test_features_endpoint():
    """Test features endpoint"""
    print("\n Testing features endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/features")
        if response.status_code == 503:
            print(f"   WARNING:  Model not loaded")
            return False
        data = response.json()
        print(f"   Total Features: {data['total_features']}")
        print(f"   Sample Features: {', '.join(data['features'][:5])}...")
        print(f"   [OK] PASSED")
        return True
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
        return False

def main():
    """Run all tests"""
    print_header(" HEA API TEST SUITE")
    print(f"Testing API at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Model Info", test_model_info()))
    results.append(("Features List", test_features_endpoint()))
    results.append(("Single Prediction", test_single_prediction()))
    results.append(("Prediction with Explanation", test_prediction_with_explanation()))
    results.append(("Batch Prediction", test_batch_prediction()))
    
    # Summary
    print_header("[DATA] TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "[FAIL] FAILED"
        print(f"{name:.<40} {status}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("All tests passed! API is working correctly.\n")
        print("Next steps:")
        print("   1. Visit http://localhost:8000/docs for interactive documentation")
        print("   2. Try uploading a CSV file via /predict/batch/upload")
        print("   3. Integrate with your web application\n")
    else:
        print("Some tests failed. Please check the output above.\n")
        if not results[1][1]:  # Model info failed
            print("Tip: Make sure model files are in models/catalog_models/ or models/final/\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user\n")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API")
        print("   Make sure the API server is running at http://localhost:8000")
        print("   Run: .\\backend\\launch_backend.ps1\n")
