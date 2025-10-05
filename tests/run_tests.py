"""
Test Runner for HEA Project
NASA Space Apps Challenge 2025

This script runs all test suites and generates a comprehensive report.
"""

import unittest
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests(verbosity=2):
    """
    Run all test suites and return results
    
    Args:
        verbosity: Test output verbosity (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        unittest.TestResult: Test results
    """
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    print("=" * 70)
    print("Running HEA Project Test Suite")
    print("NASA Space Apps Challenge 2025")
    print("=" * 70)
    print()
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print("=" * 70)
    
    return result


def run_specific_test(test_module_name, verbosity=2):
    """
    Run a specific test module
    
    Args:
        test_module_name: Name of test module (e.g., 'test_model_loading')
        verbosity: Test output verbosity
    
    Returns:
        unittest.TestResult: Test results
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module_name)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    print(f"Running {test_module_name}...")
    print("=" * 70)
    
    result = runner.run(suite)
    return result


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run HEA project tests')
    parser.add_argument(
        '--test',
        '-t',
        help='Run specific test module (e.g., test_model_loading)',
        default=None
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Quiet output'
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 2  # Default
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0
    
    # Run tests
    if args.test:
        result = run_specific_test(args.test, verbosity)
    else:
        result = run_all_tests(verbosity)
    
    # Exit with appropriate code
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
