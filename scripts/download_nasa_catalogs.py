#!/usr/bin/env python3
"""
Download NASA Exoplanet Archive catalogs.

This script downloads the latest exoplanet catalogs from NASA Exoplanet Archive
and saves them to data/catalogs/ directory.

Usage:
    python scripts/download_nasa_catalogs.py --all
    python scripts/download_nasa_catalogs.py --catalog kepler
    python scripts/download_nasa_catalogs.py --catalog tess k2
"""

import argparse
import sys
import io
from pathlib import Path
from datetime import datetime
import requests
from typing import List

# Set UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# NASA Exoplanet Archive TAP service URLs
NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Catalog definitions
CATALOGS = {
    'kepler': {
        'table': 'cumulative',
        'filename': 'kepler_koi.csv',
        'description': 'Kepler Objects of Interest (KOI)'
    },
    'k2': {
        'table': 'k2pandc',
        'filename': 'k2_catalog.csv',
        'description': 'K2 Planets and Candidates'
    },
    'tess': {
        'table': 'TOI',
        'filename': 'tess_toi.csv',
        'description': 'TESS Objects of Interest (TOI)'
    }
}


def download_catalog(catalog_name: str, output_dir: Path) -> bool:
    """
    Download a specific catalog from NASA Exoplanet Archive.
    
    Args:
        catalog_name: Name of the catalog (kepler, k2, or tess)
        output_dir: Directory to save the catalog file
        
    Returns:
        True if download was successful, False otherwise
    """
    if catalog_name not in CATALOGS:
        print(f"ERROR: Unknown catalog '{catalog_name}'. Valid options: {', '.join(CATALOGS.keys())}")
        return False
    
    catalog_info = CATALOGS[catalog_name]
    table_name = catalog_info['table']
    filename = catalog_info['filename']
    description = catalog_info['description']
    
    output_file = output_dir / filename
    
    print(f"\nDownloading {description}...")
    print(f"  Table: {table_name}")
    print(f"  Output: {output_file}")
    
    try:
        # Construct TAP query to get all columns and rows
        query = f"SELECT * FROM {table_name}"
        
        params = {
            'query': query,
            'format': 'csv'
        }
        
        # Make request with timeout
        print("  Sending request to NASA Exoplanet Archive...")
        response = requests.get(NASA_TAP_URL, params=params, timeout=300)
        response.raise_for_status()
        
        # Check if we got data
        content = response.text
        if not content or len(content) < 100:
            print(f"  ERROR: Received empty or invalid response")
            return False
        
        # Count rows (excluding header and comment lines)
        lines = content.split('\n')
        data_rows = [line for line in lines if line.strip() and not line.startswith('#')]
        row_count = len(data_rows) - 1  # Subtract header
        
        if row_count <= 0:
            print(f"  WARNING: No data rows found in response")
            return False
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Get file size
        file_size = output_file.stat().st_size / 1024  # KB
        
        print(f"  SUCCESS: Downloaded {row_count:,} rows ({file_size:.1f} KB)")
        print(f"  Saved to: {output_file}")
        
        return True
        
    except requests.exceptions.Timeout:
        print(f"  ERROR: Request timed out after 5 minutes")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Network error: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: Unexpected error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download NASA Exoplanet Archive catalogs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download all catalogs:
    python scripts/download_nasa_catalogs.py --all
    
  Download specific catalog:
    python scripts/download_nasa_catalogs.py --catalog kepler
    
  Download multiple catalogs:
    python scripts/download_nasa_catalogs.py --catalog kepler tess
    
  Save to custom directory:
    python scripts/download_nasa_catalogs.py --all --output-dir my_data/
        """
    )
    
    parser.add_argument(
        '--catalog',
        type=str,
        nargs='+',
        choices=['kepler', 'k2', 'tess'],
        help='Specific catalog(s) to download'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available catalogs'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/catalogs',
        help='Directory to save catalog files (default: data/catalogs)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.catalog:
        parser.print_help()
        print("\nERROR: Must specify either --all or --catalog")
        sys.exit(1)
    
    # Determine which catalogs to download
    if args.all:
        catalogs_to_download = list(CATALOGS.keys())
    else:
        catalogs_to_download = args.catalog
    
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("NASA Exoplanet Archive Catalog Downloader")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Catalogs: {', '.join(catalogs_to_download)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 70)
    
    # Download each catalog
    results = {}
    for catalog_name in catalogs_to_download:
        success = download_catalog(catalog_name, output_dir)
        results[catalog_name] = success
    
    # Print summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for catalog_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        symbol = "" if success else ""
        print(f"  {symbol} {CATALOGS[catalog_name]['description']:40} {status}")
    
    print("=" * 70)
    print(f"Total: {success_count}/{total_count} catalogs downloaded successfully")
    
    if success_count == total_count:
        print("\nAll catalogs downloaded successfully!")
        sys.exit(0)
    else:
        print(f"\nWARNING: {total_count - success_count} catalog(s) failed to download")
        sys.exit(1)


if __name__ == '__main__':
    main()
