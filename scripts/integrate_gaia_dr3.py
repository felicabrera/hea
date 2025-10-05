"""
Gaia DR3 Integration Script
HEA - NASA Space Apps Challenge 2025

This script cross-matches your exoplanet candidates with Gaia DR3 data
to add critical features like RUWE (unresolved binary indicator).

RUWE > 1.4 indicates unresolved binary stars - a major source of false positives!

Usage:
    python scripts/integrate_gaia_dr3.py --input data/processed/training_data.csv --output data/processed/training_data_gaia.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_astroquery():
    """Install astroquery if not available"""
    try:
        import astroquery
        return True
    except ImportError:
        logger.info("astroquery not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'astroquery'])
        logger.info(" astroquery installed successfully")
        return True


def cross_match_gaia(ra, dec, radius_arcsec=5.0):
    """
    Cross-match single target with Gaia DR3
    
    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds (default: 5")
    
    Returns:
        dict: Gaia parameters or None if no match
    """
    from astroquery.gaia import Gaia
    
    try:
        # Build query
        radius_deg = radius_arcsec / 3600.0
        query = f"""
        SELECT TOP 1 
            parallax, parallax_error,
            pmra, pmdec,
            radial_velocity,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
            bp_rp,
            ruwe,
            astrometric_excess_noise,
            phot_g_mean_flux_over_error,
            astrometric_params_solved
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        ORDER BY phot_g_mean_mag ASC
        """
        
        # Execute query
        job = Gaia.launch_job(query)
        results = job.get_results()
        
        if len(results) == 0:
            return None
        
        # Extract first result
        row = results[0]
        
        return {
            'gaia_parallax': float(row['parallax']) if row['parallax'] is not None else np.nan,
            'gaia_parallax_error': float(row['parallax_error']) if row['parallax_error'] is not None else np.nan,
            'gaia_pmra': float(row['pmra']) if row['pmra'] is not None else np.nan,
            'gaia_pmdec': float(row['pmdec']) if row['pmdec'] is not None else np.nan,
            'gaia_radial_velocity': float(row['radial_velocity']) if row['radial_velocity'] is not None else np.nan,
            'gaia_g_mag': float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] is not None else np.nan,
            'gaia_bp_mag': float(row['phot_bp_mean_mag']) if row['phot_bp_mean_mag'] is not None else np.nan,
            'gaia_rp_mag': float(row['phot_rp_mean_mag']) if row['phot_rp_mean_mag'] is not None else np.nan,
            'gaia_bp_rp': float(row['bp_rp']) if row['bp_rp'] is not None else np.nan,
            'gaia_ruwe': float(row['ruwe']) if row['ruwe'] is not None else np.nan,
            'gaia_excess_noise': float(row['astrometric_excess_noise']) if row['astrometric_excess_noise'] is not None else np.nan,
            'gaia_flux_snr': float(row['phot_g_mean_flux_over_error']) if row['phot_g_mean_flux_over_error'] is not None else np.nan,
        }
        
    except Exception as e:
        logger.warning(f"Error querying Gaia for RA={ra}, Dec={dec}: {e}")
        return None


def integrate_gaia_batch(data, ra_col='ra', dec_col='dec', batch_size=10, delay=1.0):
    """
    Integrate Gaia DR3 data for entire dataset
    
    Args:
        data: DataFrame with RA/Dec columns
        ra_col: Name of RA column (in degrees)
        dec_col: Name of Dec column (in degrees)
        batch_size: Number of queries before pause
        delay: Delay between batches in seconds
    
    Returns:
        DataFrame with Gaia columns added
    """
    logger.info("=" * 80)
    logger.info("GAIA DR3 CROSS-MATCH")
    logger.info("=" * 80)
    
    # Install astroquery if needed
    install_astroquery()
    
    # Check for RA/Dec columns
    if ra_col not in data.columns or dec_col not in data.columns:
        logger.error(f"RA column '{ra_col}' or Dec column '{dec_col}' not found!")
        logger.info(f"Available columns: {list(data.columns)}")
        return data
    
    logger.info(f"Cross-matching {len(data)} targets with Gaia DR3...")
    logger.info(f"Using columns: RA='{ra_col}', Dec='{dec_col}'")
    logger.info(f"Search radius: 5 arcseconds")
    logger.info(f"Rate limiting: {batch_size} queries per {delay}s")
    
    # Initialize Gaia columns
    gaia_columns = [
        'gaia_parallax', 'gaia_parallax_error',
        'gaia_pmra', 'gaia_pmdec', 'gaia_radial_velocity',
        'gaia_g_mag', 'gaia_bp_mag', 'gaia_rp_mag', 'gaia_bp_rp',
        'gaia_ruwe', 'gaia_excess_noise', 'gaia_flux_snr'
    ]
    
    for col in gaia_columns:
        data[col] = np.nan
    
    # Cross-match each target
    matches = 0
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Querying Gaia"):
        ra = row[ra_col]
        dec = row[dec_col]
        
        # Skip if RA/Dec are invalid
        if pd.isna(ra) or pd.isna(dec):
            continue
        
        # Query Gaia
        gaia_data = cross_match_gaia(ra, dec)
        
        if gaia_data is not None:
            matches += 1
            for col, value in gaia_data.items():
                data.at[idx, col] = value
        
        # Rate limiting
        if (idx + 1) % batch_size == 0:
            time.sleep(delay)
    
    logger.info(f"\n Cross-match complete!")
    logger.info(f"  Successful matches: {matches}/{len(data)} ({matches/len(data)*100:.1f}%)")
    
    # Analyze RUWE distribution
    if 'gaia_ruwe' in data.columns:
        valid_ruwe = data['gaia_ruwe'].dropna()
        if len(valid_ruwe) > 0:
            high_ruwe = (valid_ruwe > 1.4).sum()
            logger.info(f"\n[DATA] RUWE Analysis:")
            logger.info(f"  Valid RUWE values: {len(valid_ruwe)}")
            logger.info(f"  Mean RUWE: {valid_ruwe.mean():.3f}")
            logger.info(f"  High RUWE (>1.4): {high_ruwe} ({high_ruwe/len(valid_ruwe)*100:.1f}%)")
            logger.info(f"  → These are likely unresolved binaries (false positive candidates)")
    
    return data


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Integrate Gaia DR3 data')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with RA/Dec columns'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file with Gaia columns added'
    )
    parser.add_argument(
        '--ra-col',
        type=str,
        default='ra',
        help='Name of RA column (default: ra)'
    )
    parser.add_argument(
        '--dec-col',
        type=str,
        default='dec',
        help='Name of Dec column (default: dec)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Queries per batch (default: 10)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between batches in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.input}")
    data = pd.read_csv(args.input)
    logger.info(f" Loaded {len(data)} samples")
    
    # Integrate Gaia data
    data_gaia = integrate_gaia_batch(
        data,
        ra_col=args.ra_col,
        dec_col=args.dec_col,
        batch_size=args.batch_size,
        delay=args.delay
    )
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_gaia.to_csv(output_path, index=False)
    logger.info(f"\n Saved to: {output_path}")
    logger.info(f"  Added {len([col for col in data_gaia.columns if col.startswith('gaia_')])} Gaia columns")
    
    logger.info("\n[OK] Gaia DR3 integration complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Retrain model with new Gaia features")
    logger.info("  2. Run: python scripts/optimize_to_95_percent.py")
    logger.info("  3. RUWE filtering will automatically improve accuracy!")


if __name__ == "__main__":
    main()
