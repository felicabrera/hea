"""
Analyze Habitability of Exoplanet Candidates
NASA Space Apps Challenge 2025

Analyzes Kepler, TESS, and K2 datasets to identify
the most potentially habitable exoplanet candidates.
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Import habitability scorer
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'utils'))
from habitability_scorer import HabitabilityScorer

import warnings
warnings.filterwarnings('ignore')


def load_and_score_datasets():
    """Load all datasets and calculate habitability scores."""
    
    print("=" * 80)
    print("EXOPLANET HABITABILITY ANALYSIS")
    print("NASA Space Apps Challenge 2025")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / 'data' / 'catalogs'
    
    # Initialize scorer
    scorer = HabitabilityScorer(scoring_mode='balanced')
    
    all_results = []
    
    # ========================================
    # K2 Dataset (Most complete)
    # ========================================
    print("\n[*] Analyzing K2 Catalog...")
    try:
        k2_df = pd.read_csv(data_dir / 'k2_catalog.csv', comment='#')
        print(f"   Loaded {len(k2_df)} K2 candidates")
        
        # Score K2 dataset
        k2_scored = scorer.score_dataset(k2_df, mission='k2')
        k2_scored['mission'] = 'K2'
        k2_scored['candidate_name'] = k2_df['pl_name'].fillna(k2_df['k2_name'])
        
        # Get statistics
        valid_scores = k2_scored['hab_habitability_score'].dropna()
        print(f"   Scored: {len(valid_scores)} candidates")
        print(f"   Mean habitability: {valid_scores.mean():.3f}")
        print(f"   High habitability (>0.6): {(valid_scores >= 0.6).sum()}")
        
        all_results.append(k2_scored)
    except Exception as e:
        print(f"   [ERROR] Error loading K2: {e}")
    
    # ========================================
    # TESS Dataset
    # ========================================
    print("\n[*] Analyzing TESS TOI Catalog...")
    try:
        tess_df = pd.read_csv(data_dir / 'tess_toi.csv', comment='#', low_memory=False)
        print(f"   Loaded {len(tess_df)} TESS candidates")
        
        # Score TESS dataset
        tess_scored = scorer.score_dataset(tess_df, mission='tess')
        tess_scored['mission'] = 'TESS'
        tess_scored['candidate_name'] = tess_df['tid'].astype(str)
        
        # Get statistics
        valid_scores = tess_scored['hab_habitability_score'].dropna()
        print(f"   Scored: {len(valid_scores)} candidates")
        print(f"   Mean habitability: {valid_scores.mean():.3f}")
        print(f"   High habitability (>0.6): {(valid_scores >= 0.6).sum()}")
        
        all_results.append(tess_scored)
    except Exception as e:
        print(f"   [ERROR] Error loading TESS: {e}")
    
    # ========================================
    # Kepler Dataset (Limited parameters)
    # ========================================
    print("\n[*] Analyzing Kepler KOI Catalog...")
    try:
        kepler_df = pd.read_csv(data_dir / 'kepler_koi.csv', comment='#')
        print(f"   Loaded {len(kepler_df)} Kepler candidates")
        print(f"   [WARNING] Limited habitability parameters available")
        
        # Note: Kepler KOI catalog lacks many derived parameters
        # We'll score based on available data (mainly period and disposition)
        kepler_scored = scorer.score_dataset(kepler_df, mission='kepler')
        kepler_scored['mission'] = 'Kepler'
        kepler_scored['candidate_name'] = kepler_df['kepoi_name']
        
        # Get statistics
        valid_scores = kepler_scored['hab_habitability_score'].dropna()
        if len(valid_scores) > 0:
            print(f"   Scored: {len(valid_scores)} candidates")
            print(f"   Mean habitability: {valid_scores.mean():.3f}")
            print(f"   High habitability (>0.6): {(valid_scores >= 0.6).sum()}")
        else:
            print(f"   [WARNING] Insufficient data for habitability scoring")
        
        all_results.append(kepler_scored)
    except Exception as e:
        print(f"   [ERROR] Error loading Kepler: {e}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def generate_top_candidates_report(df: pd.DataFrame, output_file: str = 'top_habitable_candidates.csv'):
    """Generate report of top habitable candidates."""
    
    print("\n" + "=" * 80)
    print("[*] TOP POTENTIALLY HABITABLE EXOPLANETS")
    print("=" * 80)
    
    # Filter for valid scores
    df_valid = df[df['hab_habitability_score'].notna()].copy()
    
    if len(df_valid) == 0:
        print("[ERROR] No candidates with valid habitability scores found.")
        return
    
    # Sort by habitability score
    df_sorted = df_valid.sort_values('hab_habitability_score', ascending=False)
    
    # Overall statistics
    print(f"\n[STATS] OVERALL STATISTICS:")
    print(f"   Total candidates analyzed: {len(df)}")
    print(f"   Candidates with habitability scores: {len(df_valid)}")
    print(f"   Mean habitability score: {df_valid['hab_habitability_score'].mean():.3f}")
    print(f"   Median habitability score: {df_valid['hab_habitability_score'].median():.3f}")
    
    # Habitability distribution
    print(f"\n[DIST] HABITABILITY DISTRIBUTION:")
    for class_name in ['HIGH', 'MODERATE', 'LOW', 'VERY_LOW']:
        count = (df_valid['hab_habitability_class'] == class_name).sum()
        pct = 100 * count / len(df_valid)
        print(f"   {class_name:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Top 20 candidates
    top_20 = df_sorted.head(20)
    
    print(f"\n[TOP 20] MOST HABITABLE CANDIDATES:")
    print("-" * 80)
    
    for idx, row in enumerate(top_20.iterrows(), 1):
        _, data = row
        name = data.get('candidate_name', 'Unknown')
        mission = data.get('mission', 'Unknown')
        score = data['hab_habitability_score']
        esi = data.get('hab_esi', np.nan)
        hab_class = data.get('hab_habitability_class', 'Unknown')
        
        # Get planet properties
        if mission == 'K2':
            radius = data.get('pl_rade', np.nan)
            temp = data.get('pl_eqt', np.nan)
            period = data.get('pl_orbper', np.nan)
            disposition = data.get('disposition', 'Unknown')
        elif mission == 'TESS':
            radius = data.get('pl_rade', np.nan)
            temp = data.get('pl_eqt', np.nan)
            period = data.get('pl_orbper', np.nan)
            disposition = data.get('tfopwg_disp', 'Unknown')
        else:  # Kepler
            radius = data.get('koi_prad', np.nan)
            temp = data.get('koi_teq', np.nan)
            period = data.get('koi_period', np.nan)
            disposition = data.get('koi_disposition', 'Unknown')
        
        print(f"\n{idx:2d}. {name} ({mission})")
        print(f"    Habitability Score: {score:.3f} [{hab_class}]")
        if pd.notna(esi):
            print(f"    Earth Similarity:   {esi:.3f}")
        if pd.notna(radius):
            print(f"    Radius:            {radius:.2f} R⊕")
        if pd.notna(temp):
            print(f"    Temperature:       {temp:.0f} K")
        if pd.notna(period):
            print(f"    Orbital Period:    {period:.2f} days")
        print(f"    Status:            {disposition}")
    
    # Mission breakdown
    print(f"\n[MISSION] TOP 20 BY MISSION:")
    mission_counts = top_20['mission'].value_counts()
    for mission, count in mission_counts.items():
        print(f"   {mission:10s}: {count} candidates")
    
    # Export to CSV
    output_path = Path(__file__).parent.parent / output_file
    
    # Select relevant columns for export
    export_cols = [
        'candidate_name', 'mission', 
        'hab_habitability_score', 'hab_habitability_class',
        'hab_esi', 'hab_radius_score', 'hab_temperature_score',
        'hab_insolation_score', 'hab_stellar_score', 'hab_orbital_score'
    ]
    
    # Add mission-specific columns
    if 'pl_rade' in df_sorted.columns:
        export_cols.extend(['pl_rade', 'pl_eqt', 'pl_orbper', 'pl_insol', 'st_teff'])
    if 'koi_prad' in df_sorted.columns:
        export_cols.extend(['koi_period', 'koi_disposition'])
    
    # Filter existing columns
    export_cols = [col for col in export_cols if col in df_sorted.columns]
    
    df_export = df_sorted[export_cols].head(100)  # Top 100
    df_export.to_csv(output_path, index=False)
    
    print(f"\n[SAVED] Exported top 100 candidates to: {output_path}")
    
    # Generate habitability insights
    print(f"\n[INSIGHTS] HABITABILITY INSIGHTS:")
    
    # Confirmed vs Candidates
    if 'disposition' in df_valid.columns or 'koi_disposition' in df_valid.columns:
        disp_col = 'disposition' if 'disposition' in df_valid.columns else 'koi_disposition'
        confirmed = df_valid[df_valid[disp_col] == 'CONFIRMED']
        if len(confirmed) > 0:
            print(f"   Confirmed habitable planets (score >0.6): {(confirmed['hab_habitability_score'] >= 0.6).sum()}")
            print(f"   Best confirmed candidate: {confirmed.iloc[0].get('candidate_name', 'Unknown')} "
                  f"(score: {confirmed.iloc[0]['hab_habitability_score']:.3f})")
    
    # Radius analysis
    if 'hab_radius_score' in df_valid.columns:
        earth_like = df_valid[df_valid['hab_radius_score'] >= 0.9]
        print(f"   Earth-sized planets (0.8-1.2 R⊕): {len(earth_like)}")
    
    # Temperature analysis
    if 'hab_temperature_score' in df_valid.columns:
        temperate = df_valid[df_valid['hab_temperature_score'] >= 0.8]
        print(f"   Temperate planets (good temperature): {len(temperate)}")
    
    # Goldilocks candidates (high on all metrics)
    goldilocks = df_valid[
        (df_valid['hab_radius_score'] >= 0.8) &
        (df_valid['hab_temperature_score'] >= 0.8) &
        (df_valid['hab_insolation_score'] >= 0.8)
    ]
    print(f"   'Goldilocks' candidates (high on all metrics): {len(goldilocks)}")


def main():
    """Main analysis pipeline."""
    
    # Load and score datasets
    df_all = load_and_score_datasets()
    
    if len(df_all) == 0:
        print("\n[ERROR] No data loaded. Please check catalog files.")
        return
    
    # Generate report
    generate_top_candidates_report(df_all)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
