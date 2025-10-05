"""
Test Suite for Habitable Exoplanet Analyzer (HEA)
NASA Space Apps Challenge 2025

This test suite provides comprehensive coverage of the HEA project,
including model validation, habitability analysis, data processing,
API endpoints, and integration testing.

Test Modules:
- test_model_loading: ML model loading and prediction tests
- test_habitability_scorer: Habitability scoring system tests
- test_data_loading: Data file and catalog loading tests
- test_api_endpoints: FastAPI backend endpoint tests
- test_preprocessing: Data preprocessing and feature engineering tests
- test_integration: End-to-end integration tests

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --test <module>    # Run specific module
    python -m unittest discover tests -v         # Alternative runner
"""

__version__ = "1.0.0"
__author__ = "HEA Development Team"
__status__ = "Production"
