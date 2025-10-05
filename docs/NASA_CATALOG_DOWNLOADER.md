# NASA Catalog Downloader

This script downloads the latest exoplanet catalogs from NASA Exoplanet Archive.

## Features

- Downloads catalogs directly from NASA TAP service
- Saves to `data/catalogs/` directory
- Supports Kepler, K2, and TESS missions
- Can download individual catalogs or all at once
- Shows download progress and statistics

## Usage

### Download All Catalogs

```bash
python scripts/download_nasa_catalogs.py --all
```

This will download:
- `data/catalogs/kepler_koi.csv` - Kepler Objects of Interest
- `data/catalogs/k2_catalog.csv` - K2 Planets and Candidates  
- `data/catalogs/tess_toi.csv` - TESS Objects of Interest

### Download Specific Catalog

```bash
# Kepler only
python scripts/download_nasa_catalogs.py --catalog kepler

# TESS only
python scripts/download_nasa_catalogs.py --catalog tess

# K2 only
python scripts/download_nasa_catalogs.py --catalog k2
```

### Download Multiple Catalogs

```bash
python scripts/download_nasa_catalogs.py --catalog kepler tess
```

### Custom Output Directory

```bash
python scripts/download_nasa_catalogs.py --all --output-dir my_custom_folder/
```

## Output Files

The script creates CSV files with all available columns from NASA Exoplanet Archive:

- **kepler_koi.csv**: ~9,500 rows with 134+ columns including orbital parameters, stellar properties, and disposition
- **k2_catalog.csv**: ~450 rows with 280+ columns including planet and host star data
- **tess_toi.csv**: ~7,000 rows with 150+ columns including TOI parameters and validation status

## From Webapp

You can also download catalogs from the webapp:
1. Go to "Train Model" tab
2. Click "Update NASA Dataset" section
3. Select data source (Kepler, TESS, K2, or All)
4. Click "Update Dataset" button

## Requirements

- Python 3.7+
- `requests` library (pip install requests)
- Internet connection
- ~5-10 MB disk space for all catalogs

## Troubleshooting

**Timeout Error**: If download times out, try downloading individual catalogs instead of all at once.

**Network Error**: Check your internet connection and firewall settings. NASA TAP service must be accessible.

**Empty Response**: NASA servers might be temporarily unavailable. Wait a few minutes and try again.

## NASA Exoplanet Archive

Data source: https://exoplanetarchive.ipac.caltech.edu/

The NASA Exoplanet Archive is operated by Caltech under contract with NASA.
