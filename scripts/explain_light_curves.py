#!/usr/bin/env python3
"""
Example script showing what light curves look like and how they reveal exoplanets.

This creates a synthetic example to demonstrate the concept.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_synthetic_transit():
    """Create a synthetic transit light curve to demonstrate the concept."""
    
    # Time array (30 days of observations, every 30 minutes like Kepler)
    time = np.linspace(0, 30, 1440)  # 30 days, 30-min cadence
    
    # Base star brightness (normalized to 1.0)
    flux = np.ones_like(time)
    
    # Add some stellar noise/variability
    np.random.seed(42)
    flux += 0.001 * np.random.normal(0, 1, len(time))  # 0.1% noise
    
    # Add periodic stellar variability (star rotation)
    flux += 0.002 * np.sin(2 * np.pi * time / 25)  # 25-day rotation period
    
    # Add exoplanet transits
    planet_period = 3.5  # days
    transit_duration = 3.0 / 24  # 3 hours
    transit_depth = 0.01  # 1% transit depth (Earth-like)
    
    # Calculate transit times
    transit_times = np.arange(1.0, 30, planet_period)  # Every 3.5 days
    
    for t_center in transit_times:
        # Create transit shape (simplified)
        transit_mask = np.abs(time - t_center) < transit_duration/2
        flux[transit_mask] *= (1 - transit_depth)
    
    return time, flux, transit_times

def plot_light_curve():
    """Create example light curve plots."""
    
    time, flux, transit_times = create_synthetic_transit()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full light curve
    ax1.plot(time, flux, 'b.', markersize=1, alpha=0.7)
    ax1.set_ylabel('Relative Flux')
    ax1.set_title('[STAR] Exoplanet Transit Light Curve - Full 30 Days')
    ax1.grid(True, alpha=0.3)
    
    # Mark transit times
    for t_transit in transit_times:
        ax1.axvline(t_transit, color='red', alpha=0.5, linestyle='--', linewidth=1)
    
    # Zoomed view of one transit
    zoom_center = transit_times[2]  # Third transit
    zoom_range = 0.5  # ±12 hours
    zoom_mask = np.abs(time - zoom_center) < zoom_range
    
    ax2.plot(time[zoom_mask], flux[zoom_mask], 'bo-', markersize=3, linewidth=1)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Relative Flux')
    ax2.set_title('🪐 Zoomed View: Single Transit Event')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate('Planet starts\ntransit', 
                xy=(zoom_center - 0.06, 0.995), 
                xytext=(zoom_center - 0.3, 1.002),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax2.annotate('Transit\nminimum', 
                xy=(zoom_center, 0.99), 
                xytext=(zoom_center + 0.2, 0.985),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "light_curve_example.png", dpi=150, bbox_inches='tight')
    
    print("[DATA] Light curve example saved to: docs/light_curve_example.png")
    print("\n[SEARCH] What you see in this plot:")
    print("- Normal star brightness oscillates around 1.0 (100%)")
    print("- Every 3.5 days, brightness drops by 1% for ~3 hours")
    print("- This periodic dimming reveals a planet orbiting the star!")
    print("- ML models learn to detect these subtle patterns")
    
    plt.show()

def analyze_transit_properties():
    """Show how transit properties reveal planet characteristics."""
    
    print("\n🪐 WHAT LIGHT CURVES TELL US ABOUT EXOPLANETS:")
    print("=" * 60)
    
    print("\n1.  PLANET SIZE (Transit Depth)")
    print("   - Deeper transit = Larger planet")
    print("   - Earth: ~0.01% (around Sun-like star)")
    print("   - Jupiter: ~1% (around Sun-like star)")
    print("   - Super-Earth: ~0.05%")
    
    print("\n2.  ORBITAL PERIOD (Time Between Transits)")
    print("   - Earth: 365 days")
    print("   - Hot Jupiter: 1-10 days")
    print("   - Habitable zone planets: 100-400 days")
    
    print("\n3. ⏱ PLANET DISTANCE (Transit Duration)")
    print("   - Longer transit = Closer to star OR larger orbit")
    print("   - Combined with period gives orbital distance")
    
    print("\n4.  TEMPERATURE (From orbital distance)")
    print("   - Closer planets are hotter")
    print("   - Habitable zone: liquid water possible")
    
    print("\n5.  ATMOSPHERE (Advanced analysis)")
    print("   - Transit spectroscopy reveals atmospheric composition")
    print("   - Water vapor, methane, oxygen signatures")

if __name__ == '__main__':
    print("[STAR] UNDERSTANDING EXOPLANET LIGHT CURVES")
    print("=" * 50)
    
    # Create example plots
    plot_light_curve()
    
    # Explain the science
    analyze_transit_properties()
    
    print(f"\n[START] YOUR ML MISSION:")
    print("- Train models to automatically detect these transit patterns")
    print("- Use 21,271 real exoplanet candidates from NASA")
    print("- Win the NASA Space Apps Challenge! [SUCCESS]")