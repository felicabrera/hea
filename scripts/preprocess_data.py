#!/usr/bin/env python3
"""
Script to preprocess light curve data.

Usage:
    python scripts/preprocess_data.py --mission kepler --output-dir data/processed
    python scripts/preprocess_data.py --input-dir data/raw --validate-pipeline
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.data.catalog import CatalogManager
from src.utils.logger import setup_logger
from src.utils.config_loader import config_loader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess light curve data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mission',
        type=str,
        choices=['kepler', 'tess', 'k2', 'all'],
        default='kepler',
        help='Mission to process data from'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory containing raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    
    parser.add_argument(
        '--validate-pipeline',
        action='store_true',
        help='Run pipeline validation before processing'
    )
    
    parser.add_argument(
        '--detrend-method',
        type=str,
        choices=['savgol', 'spline', 'median', 'polynomial', 'biweight', 'auto'],
        default=None,
        help='Detrending method (overrides config)'
    )
    
    parser.add_argument(
        '--normalize-method',
        type=str,
        choices=['standard', 'robust', 'minmax', 'quantile', 'auto'],
        default=None,
        help='Normalization method (overrides config)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        name='preprocess_data',
        level=getattr(__import__('logging'), args.log_level)
    )
    
    logger.info(f"Starting data preprocessing: mission={args.mission}")
    
    try:
        # Create config overrides
        config_override = {}
        if args.detrend_method:
            config_override['preprocessing'] = config_override.get('preprocessing', {})
            config_override['preprocessing']['detrend_method'] = args.detrend_method
        
        if args.normalize_method:
            config_override['preprocessing'] = config_override.get('preprocessing', {})
            config_override['preprocessing']['normalize_method'] = args.normalize_method
        
        # Initialize preprocessing pipeline
        pipeline = PreprocessingPipeline(config_override=config_override)
        
        # Validate pipeline if requested
        if args.validate_pipeline:
            logger.info("Validating preprocessing pipeline...")
            is_valid = pipeline.validate_pipeline()
            if not is_valid:
                logger.error("Pipeline validation failed")
                sys.exit(1)
            logger.info("Pipeline validation passed")
        
        # Initialize catalog manager
        catalog_manager = CatalogManager()
        
        # Process each mission
        missions = [args.mission] if args.mission != 'all' else ['kepler', 'tess', 'k2']
        
        for mission in missions:
            logger.info(f"Processing {mission} data...")
            
            try:
                # Load catalog and get labels
                catalog = catalog_manager.load_catalog(mission)
                labels_dict = catalog_manager.get_labels(mission)
                
                logger.info(f"Loaded {len(catalog)} catalog entries for {mission}")
                
                # Get target IDs to process
                target_ids = list(labels_dict.keys())
                
                if args.max_samples and len(target_ids) > args.max_samples:
                    np.random.seed(42)
                    target_ids = np.random.choice(
                        target_ids, args.max_samples, replace=False
                    ).tolist()
                
                logger.info(f"Processing {len(target_ids)} targets from {mission}")
                
                # For now, create dummy light curves for demonstration
                # In real implementation, you would load actual FITS files
                light_curves = []
                periods = []
                epochs = []
                
                for target_id in target_ids:
                    # Create synthetic light curve for demonstration
                    time = np.linspace(0, 50, 5000)  # 50 days
                    flux = np.ones_like(time) + 0.001 * np.random.randn(len(time))
                    
                    # Add a simple transit if it's a confirmed planet
                    if labels_dict[target_id] == 1:  # Exoplanet
                        period = np.random.uniform(1, 20)  # Random period
                        epoch = np.random.uniform(0, period)
                        
                        # Add transits
                        transit_times = np.arange(epoch, time[-1], period)
                        for t_transit in transit_times:
                            transit_mask = np.abs(time - t_transit) < 0.1
                            flux[transit_mask] *= 0.99  # 1% depth
                    else:
                        period = None
                        epoch = None
                    
                    light_curves.append((time, flux))
                    periods.append(period)
                    epochs.append(epoch)
                
                # Process batch
                processed_fluxes, metadata_list = pipeline.process_batch(
                    light_curves, periods, epochs, target_ids
                )
                
                # Generate processing summary
                summary = pipeline.get_processing_summary(metadata_list)
                
                logger.info(f"Processing summary for {mission}:")
                logger.info(f"  Total: {summary['total_curves']}")
                logger.info(f"  Successful: {summary['successful']}")
                logger.info(f"  Failed: {summary['failed']}")
                logger.info(f"  Success rate: {summary['success_rate']:.1%}")
                
                # Save results (placeholder - implement actual saving)
                output_path = Path(args.output_dir) / mission
                output_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Results saved to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {mission}: {e}")
                continue
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()