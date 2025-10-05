# Test Suite Documentation
## NASA Space Apps Challenge 2025 - HEA Project

This directory contains a comprehensive test suite for the Habitable Exoplanet Analyzer project. The tests are designed to ensure code quality, reliability, and maintainability for NASA-grade standards.

## Test Coverage

### 1. Model Loading Tests (`test_model_loading.py`)
Tests the machine learning model loading and prediction functionality:
- Model file existence and loading
- Prediction methods (predict, predict_proba)
- Model metadata validation
- Prediction output validation
- Performance thresholds (accuracy ≥90%, recall ≥95%, precision ≥90%, F1 ≥90%, AUC ≥95%)

**15 test cases** covering model lifecycle and performance validation.

### 2. Habitability Scorer Tests (`test_habitability_scorer.py`)
Tests the habitability scoring system:
- Scorer initialization and configuration
- Earth-like planet scoring (high scores expected)
- Hot Jupiter scoring (low scores expected)
- Score range validation (0.0 to 1.0)
- None value handling
- Extreme value handling
- Component score calculation
- Reproducibility verification
- Classification logic (HIGH/MEDIUM/LOW)

**12 test cases** covering habitability analysis workflows.

### 3. Data Loading Tests (`test_data_loading.py`)
Tests data file structure and catalog loading:
- Data directory structure validation
- Kepler, TESS, and K2 catalog file existence
- Catalog loading and parsing
- Required column validation (mission-specific)
- Data quality checks (non-empty, valid structure)
- Habitability rankings file validation

**17 test cases** covering data infrastructure.

### 4. API Endpoint Tests (`test_api_endpoints.py`)
Tests the FastAPI backend endpoints:
- Health check and root endpoints
- Single prediction endpoint validation
- Batch prediction endpoint
- Habitability analysis endpoint
- Input validation (missing fields, invalid types)
- Edge case handling (negative values, extreme values)
- Probability range validation
- Earth-like vs. hot Jupiter scoring via API

**18 test cases** covering API functionality (requires FastAPI installed).

### 5. Preprocessing Tests (`test_preprocessing.py`)
Tests data preprocessing and feature engineering:
- Feature engineering functions
- Required column validation
- Numeric data type validation
- Habitability zone calculations
- Planet size classification
- Stellar type inference
- Insolation flux normalization
- Missing value detection
- Negative value detection
- Data transformations (log, standardization, min-max scaling)
- Outlier detection (IQR method)
- Feature selection (variance, correlation)

**23 test cases** covering data preprocessing pipeline.

### 6. Integration Tests (`test_integration.py`)
Tests end-to-end workflows and component interactions:
- Single planet prediction pipeline
- Batch prediction pipeline
- Probability prediction pipeline
- Habitability scoring pipeline
- Habitability classification pipeline
- Catalog loading and processing pipeline
- Model persistence (save/load)
- Model metadata validation
- API integration tests (prediction and habitability endpoints)
- Error handling tests

**15 test cases** covering complete system workflows.

## Total Test Coverage
**100+ test cases** across 6 test modules covering:
- Machine learning model operations
- Habitability analysis
- Data loading and validation
- API endpoints
- Data preprocessing
- End-to-end integration

## Running Tests

### Run All Tests
```powershell
# From project root
python tests/run_tests.py

# Or using unittest directly
python -m unittest discover tests -v
```

### Run Specific Test Module
```powershell
# Using test runner
python tests/run_tests.py --test test_model_loading

# Using unittest directly
python -m unittest tests.test_model_loading -v
```

### Run Specific Test Class
```powershell
python -m unittest tests.test_model_loading.TestModelLoading -v
```

### Run Specific Test Method
```powershell
python -m unittest tests.test_model_loading.TestModelLoading.test_model_file_exists -v
```

### Verbosity Options
```powershell
# Verbose output (default)
python tests/run_tests.py --verbose

# Quiet output (only show failures)
python tests/run_tests.py --quiet
```

## Test Requirements

### Core Requirements
- Python 3.8+
- pandas
- numpy
- joblib (for model loading)

### Optional Requirements
- FastAPI (for API endpoint tests)
- pytest (alternative test runner)

Install all requirements:
```powershell
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

## Test Structure

Each test file follows this structure:
```python
import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestClassName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One-time setup for all tests in class"""
        pass
    
    def setUp(self):
        """Setup before each test method"""
        pass
    
    def test_something(self):
        """Test description"""
        self.assertEqual(actual, expected)
    
    def tearDown(self):
        """Cleanup after each test method"""
        pass
```

## Continuous Integration

These tests are designed to integrate with CI/CD pipelines:

### GitHub Actions Example
```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r backend/requirements.txt
      - name: Run tests
        run: python tests/run_tests.py
```

## Best Practices

1. **Test Isolation**: Each test is independent and doesn't rely on other tests
2. **Clear Naming**: Test names clearly describe what is being tested
3. **Comprehensive Coverage**: Tests cover happy paths, edge cases, and error conditions
4. **Performance Tests**: Model performance thresholds validated
5. **Documentation**: Each test has a docstring explaining its purpose
6. **Skip Conditions**: Tests gracefully skip when dependencies are unavailable

## Adding New Tests

When adding new features, create corresponding tests:

1. Create new test file: `test_new_feature.py`
2. Follow existing structure and naming conventions
3. Include docstrings for all test methods
4. Add both positive and negative test cases
5. Run tests to ensure they pass
6. Update this README with test coverage information

## NASA Space Apps Challenge Standards

These tests meet NASA Space Apps Challenge requirements:
- ✅ Comprehensive test coverage (100+ test cases)
- ✅ Professional code quality
- ✅ Scalable architecture
- ✅ Edge case handling
- ✅ Performance validation
- ✅ Error handling
- ✅ Documentation
- ✅ CI/CD ready

## Troubleshooting

### Tests Skipped
Some tests may be skipped if dependencies are not installed:
- **FastAPI tests**: Install `pip install fastapi[all]`
- **Model tests**: Ensure model file exists in `models/` directory
- **Habitability tests**: Ensure `src/utils/habitability_scorer.py` exists

### Import Errors
If you encounter import errors:
```powershell
# Ensure project root is in Python path
$env:PYTHONPATH="C:\Users\feli\Dev\hea"
python tests/run_tests.py
```

### Test Failures
If tests fail:
1. Check that all dependencies are installed
2. Ensure data files exist in `data/catalogs/`
3. Verify model file exists in `models/`
4. Check log files for detailed error messages

## Contact

For questions about the test suite, please refer to the main project README.md or open an issue on GitHub.
