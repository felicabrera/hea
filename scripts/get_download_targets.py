#!/usr/bin/env python3
"""
Script to extract target IDs for light curve downloads from catalogs.

Usage:
    python scripts/get_download_targets.py --mission kepler --max-targets 100
    python scripts/get_download_targets.py --confirmed-only --all-missions
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.catalog import CatalogManager
from src.utils.logger import setup_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Get target IDs for light curve downloads',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mission',
        type=str,
        choices=['kepler', 'tess', 'k2'],
        default='kepler',
        help='Mission to get targets from'
    )
    
    parser.add_argument(
        '--max-targets',
        type=int,
        default=100,
        help='Maximum number of targets to extract'
    )
    
    parser.add_argument(
        '--confirmed-only',
        action='store_true',
        help='Only get confirmed exoplanets'
    )
    
    parser.add_argument(
        '--all-missions',
        action='store_true',
        help='Get targets from all missions'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Save target list to file'
    )
    
    return parser.parse_args()

def get_targets_for_mission(cm: CatalogManager, mission: str, max_targets: int, confirmed_only: bool, logger):
    """Get target IDs for a specific mission."""
    
    logger.info(f"Getting targets for {mission.upper()}...")
    
    try:
        catalog = cm.load_catalog(mission)
        logger.info(f"Loaded {len(catalog)} catalog entries")
        
        targets = []
        
        if mission.lower() == 'kepler':
            # Kepler KOI format
            for _, row in catalog.iterrows():
                if confirmed_only and row['koi_disposition'] != 'CONFIRMED':
                    continue
                
                kic_id = int(row['kepid'])
                target_info = {
                    'mission': 'kepler',
                    'target_id': f'KIC {kic_id}',
                    'kic_id': kic_id,
                    'koi_name': row['kepoi_name'],
                    'disposition': row['koi_disposition'],
                    'period': row.get('koi_period', 'Unknown'),
                    'download_search': f'KIC {kic_id}',
                    'filename_pattern': f'kic_{kic_id:08d}_kepler.fits'
                }
                targets.append(target_info)
                
                if len(targets) >= max_targets:
                    break
                    
        elif mission.lower() == 'tess':
            # TESS TOI format
            for _, row in catalog.iterrows():
                if confirmed_only and row['tfopwg_disp'] not in ['CP', 'KP']:  # Confirmed Planet, Known Planet
                    continue
                
                tic_id = int(row['tid'])
                target_info = {
                    'mission': 'tess',
                    'target_id': f'TIC {tic_id}',
                    'tic_id': tic_id,
                    'toi_name': row['toi'],
                    'disposition': row['tfopwg_disp'],
                    'period': row.get('pl_orbper', 'Unknown'),
                    'download_search': f'TIC {tic_id}',
                    'filename_pattern': f'tic_{tic_id:09d}_tess.fits'
                }
                targets.append(target_info)
                
                if len(targets) >= max_targets:
                    break
                    
        elif mission.lower() == 'k2':
            # K2 format (varies, need to check column names)
            logger.warning("K2 target extraction not fully implemented - check catalog column names")
            
        logger.info(f"[OK] Found {len(targets)} targets for {mission.upper()}")
        return targets
        
    except Exception as e:
        logger.error(f"[FAIL] Error getting targets for {mission}: {e}")
        return []

def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(name='get_targets', level='INFO')
    
    cm = CatalogManager()
    
    if args.all_missions:
        missions = ['kepler', 'tess']  # Skip k2 for now
    else:
        missions = [args.mission]
    
    all_targets = []
    
    for mission in missions:
        targets = get_targets_for_mission(cm, mission, args.max_targets, args.confirmed_only, logger)
        all_targets.extend(targets)
    
    if not all_targets:
        logger.error("No targets found!")
        return
    
    # Display results
    logger.info(f"\n[TARGET] DOWNLOAD TARGET LIST ({len(all_targets)} targets)")
    logger.info("="*60)
    
    for i, target in enumerate(all_targets[:20]):  # Show first 20
        logger.info(f"{i+1:2d}. {target['target_id']} - {target['disposition']} - Period: {target['period']}")
        logger.info(f"    Search: {target['download_search']}")
        logger.info(f"    Save as: {target['filename_pattern']}")
        logger.info("")
    
    if len(all_targets) > 20:
        logger.info(f"... and {len(all_targets) - 20} more targets")
    
    # Save to file if requested
    if args.output_file:
        df = pd.DataFrame(all_targets)
        df.to_csv(args.output_file, index=False)
        logger.info(f" Saved target list to: {args.output_file}")
    
    # Download instructions
    logger.info("\n DOWNLOAD INSTRUCTIONS:")
    logger.info("="*60)
    logger.info("1. Go to: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html")
    logger.info("2. Search for each target ID (e.g., 'KIC 757076')")
    logger.info("3. Click target → 'Data Products' → Filter 'Time Series' → 'Light curves'")
    logger.info("4. Download .fits files")
    logger.info("5. Save to appropriate directory:")
    for mission in set(t['mission'] for t in all_targets):
        logger.info(f"   - {mission.upper()}: data/raw/{mission}/")
    
    logger.info(f"\n[START] Ready to download {len(all_targets)} light curves!")

if __name__ == '__main__':
    main()