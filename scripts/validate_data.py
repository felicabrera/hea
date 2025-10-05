#!/usr/bin/env python3
"""
Script to validate manually downloaded data.

Usage:
    python scripts/validate_data.py --mission kepler
    python scripts/validate_data.py --check-all
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.catalog import CatalogManager
from src.utils.logger import setup_logger
from src.utils.config_loader import config_loader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate manually downloaded exoplanet data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mission',
        type=str,
        choices=['kepler', 'tess', 'k2'],
        default='kepler',
        help='Mission to validate data for'
    )
    
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Check all missions'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Root data directory'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()

def validate_mission_data(mission: str, data_dir: Path, logger) -> dict:
    """Validate data for a specific mission."""
    results = {
        'mission': mission,
        'catalog_exists': False,
        'catalog_entries': 0,
        'lightcurve_dir_exists': False,
        'lightcurve_files': 0,
        'status': 'missing'
    }
    
    # Check catalog file
    catalog_file = data_dir / 'catalogs' / f'{mission}_{"koi" if mission == "kepler" else "toi" if mission == "tess" else "catalog"}.csv'
    if catalog_file.exists():
        results['catalog_exists'] = True
        try:
            # NASA Exoplanet Archive CSV files have comment headers starting with #
            # We need to skip these lines when reading
            catalog = pd.read_csv(catalog_file, comment='#')
            results['catalog_entries'] = len(catalog)
            logger.info(f"[OK] {mission.upper()} catalog: {results['catalog_entries']} entries")
        except Exception as e:
            logger.error(f"[FAIL] Error reading {mission} catalog: {e}")
    else:
        logger.warning(f"WARNING:  {mission.upper()} catalog not found: {catalog_file}")
    
    # Check light curve directory
    lc_dir = data_dir / 'raw' / mission
    if lc_dir.exists():
        results['lightcurve_dir_exists'] = True
        fits_files = list(lc_dir.glob('*.fits'))
        results['lightcurve_files'] = len(fits_files)
        if results['lightcurve_files'] > 0:
            logger.info(f"[OK] {mission.upper()} light curves: {results['lightcurve_files']} files")
            results['status'] = 'complete'
        else:
            logger.warning(f"WARNING:  {mission.upper()} light curve directory empty")
            results['status'] = 'partial'
    else:
        logger.warning(f"WARNING:  {mission.upper()} light curve directory not found: {lc_dir}")
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        name='validate_data',
        level=getattr(__import__('logging'), args.log_level)
    )
    
    data_dir = Path(args.data_dir)
    
    if args.check_all:
        missions = ['kepler', 'tess', 'k2']
        logger.info("Validating data for all missions...")
    else:
        missions = [args.mission]
        logger.info(f"Validating data for {args.mission}...")
    
    try:
        all_results = []
        
        for mission in missions:
            logger.info(f"\n[DATA] Checking {mission.upper()} data...")
            results = validate_mission_data(mission, data_dir, logger)
            all_results.append(results)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info(" DATA VALIDATION SUMMARY")
        logger.info("="*50)
        
        for results in all_results:
            status_emoji = "[OK]" if results['status'] == 'complete' else "WARNING:" if results['status'] == 'partial' else "[FAIL]"
            logger.info(f"{status_emoji} {results['mission'].upper()}: {results['catalog_entries']} catalog entries, {results['lightcurve_files']} light curves")
        
        # Instructions for missing data
        missing_missions = [r for r in all_results if r['status'] == 'missing']
        if missing_missions:
            logger.info("\n TO DOWNLOAD MISSING DATA:")
            logger.info("1. Create directories: data/catalogs/ and data/raw/{kepler,tess,k2}/")
            logger.info("2. Download catalogs from NASA Exoplanet Archive:")
            for results in missing_missions:
                mission = results['mission']
                if mission == 'kepler':
                    logger.info("   - Kepler KOI: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi")
                elif mission == 'tess':
                    logger.info("   - TESS TOI: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=toi")
                elif mission == 'k2':
                    logger.info("   - K2: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates")
            logger.info("3. Download light curves from MAST: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html")
        
        complete_missions = [r for r in all_results if r['status'] == 'complete']
        if complete_missions:
            logger.info(f"\n[START] Ready to process {len(complete_missions)} mission(s)!")
            logger.info("Next step: python scripts/preprocess_data.py")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()