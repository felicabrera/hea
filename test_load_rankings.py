"""
Quick test to verify habitability rankings loading
"""
from pathlib import Path
import pandas as pd

def load_top_habitable_candidates():
    """Load pre-computed top habitable candidates."""
    try:
        # Get absolute path to project root
        project_root = Path(__file__).parent.absolute()
        
        # Check both possible locations using absolute paths
        csv_paths = [
            project_root / "top_habitable_candidates.csv",  # Root (where script saves)
            project_root / "data" / "results" / "top_habitable_candidates.csv"  # Expected location
        ]
        
        for csv_path in csv_paths:
            print(f"Checking: {csv_path}")
            print(f"  Exists: {csv_path.exists()}")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"   Loaded {len(df)} rows successfully!")
                return df
        print("   No file found")
        return None
    except Exception as e:
        # Print error for debugging
        print(f"Error loading habitability data: {e}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("Testing Habitability Rankings Loader")
    print("="*60)
    
    df = load_top_habitable_candidates()
    
    if df is not None:
        print(f"\n[OK] SUCCESS: Loaded {len(df)} candidates")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 3 candidates:")
        print(df[['candidate_name', 'mission', 'hab_habitability_score', 'hab_habitability_class']].head(3))
    else:
        print("\n[FAIL] FAILED: Could not load data")
