"""
Weather Simulation Module.
Simulates flood conditions based on humidity data from weather station.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime

from .config import get_humidity_weight
from .data_loader import load_weather_data


def get_current_weather_simulation() -> Dict:
    """
    Get simulated current weather conditions based on station data.
    
    Returns:
        Dictionary with weather conditions
    """
    weather_df = load_weather_data()
    
    if 'Humidity' in weather_df.columns:
        avg_humidity = weather_df['Humidity'].mean()
        max_humidity = weather_df['Humidity'].max()
        min_humidity = weather_df['Humidity'].min()
    else:
        # Default values
        avg_humidity = 75.0
        max_humidity = 95.0
        min_humidity = 65.0
    
    if 'Rainfall' in weather_df.columns:
        avg_rainfall = weather_df['Rainfall'].mean()
        max_rainfall = weather_df['Rainfall'].max()
    else:
        avg_rainfall = 0.0
        max_rainfall = 0.0
    
    humidity_weight, humidity_label = get_humidity_weight(avg_humidity)
    
    return {
        'humidity': {
            'avg': avg_humidity,
            'max': max_humidity,
            'min': min_humidity,
            'weight': humidity_weight,
            'label': humidity_label
        },
        'rainfall': {
            'avg': avg_rainfall,
            'max': max_rainfall
        },
        'records': len(weather_df)
    }


def simulate_normal_condition() -> Tuple[float, str]:
    """
    Simulate normal weather condition (low flood risk).
    
    Returns:
        Tuple of (humidity, description)
    """
    # Normal humidity (≤ 70%)
    humidity = np.random.uniform(55, 70)
    return humidity, "Kondisi Normal - Risiko Banjir Rendah"


def simulate_low_risk_condition() -> Tuple[float, str]:
    """
    Simulate low risk condition.
    
    Returns:
        Tuple of (humidity, description)
    """
    # Low risk humidity (71-80%)
    humidity = np.random.uniform(71, 80)
    return humidity, "Potensi Hujan Ringan - Risiko Banjir Rendah"


def simulate_medium_risk_condition() -> Tuple[float, str]:
    """
    Simulate medium risk condition.
    
    Returns:
        Tuple of (humidity, description)
    """
    # Medium risk humidity (81-90%)
    humidity = np.random.uniform(81, 90)
    return humidity, "Potensi Hujan Sedang - Risiko Banjir Sedang"


def simulate_high_risk_condition() -> Tuple[float, str]:
    """
    Simulate high flood risk condition.
    
    Returns:
        Tuple of (humidity, description)
    """
    # High risk humidity (91-95%)
    humidity = np.random.uniform(91, 95)
    return humidity, "Potensi Hujan Lebat - Risiko Banjir Tinggi"


def simulate_very_high_risk_condition() -> Tuple[float, str]:
    """
    Simulate very high flood risk condition.
    
    Returns:
        Tuple of (humidity, description)
    """
    # Very high risk humidity (> 95%)
    humidity = np.random.uniform(96, 100)
    return humidity, "Potensi Hujan Sangat Lebat - Risiko Banjir Sangat Tinggi"


def get_simulation_scenario(scenario: str) -> Tuple[float, str]:
    """
    Get weather simulation based on scenario name.
    
    Args:
        scenario: One of 'normal', 'low', 'medium', 'high', 'very_high', 'random'
    
    Returns:
        Tuple of (humidity, description)
    """
    scenarios = {
        'normal': simulate_normal_condition,
        'low': simulate_low_risk_condition,
        'medium': simulate_medium_risk_condition,
        'high': simulate_high_risk_condition,
        'very_high': simulate_very_high_risk_condition
    }
    
    if scenario == 'random':
        # Random scenario
        scenario = np.random.choice(list(scenarios.keys()))
    
    if scenario in scenarios:
        return scenarios[scenario]()
    else:
        return simulate_normal_condition()


def generate_random_origin(
    flood_points: pd.DataFrame,
    evacuation_points: pd.DataFrame
) -> Tuple[float, float]:
    """
    Generate random origin point for simulation.
    
    Args:
        flood_points: DataFrame with flood locations
        evacuation_points: DataFrame with evacuation locations
    
    Returns:
        Tuple of (latitude, longitude)
    """
    # Combine all points to get area bounds
    all_lats = list(flood_points['Latitude']) + list(evacuation_points['Latitude'])
    all_lons = list(flood_points['Longitude']) + list(evacuation_points['Longitude'])
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Generate random point within bounds
    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)
    
    return (lat, lon)


if __name__ == "__main__":
    # Test weather simulation
    weather = get_current_weather_simulation()
    print("Current Weather Simulation:")
    print(f"  Humidity: {weather['humidity']['avg']:.1f}% (Range: {weather['humidity']['min']:.1f}-{weather['humidity']['max']:.1f}%)")
    print(f"  Flood Weight: {weather['humidity']['weight']} ({weather['humidity']['label']})")
    print(f"  Rainfall Avg: {weather['rainfall']['avg']:.1f} mm")
    
    print("\nScenario Tests:")
    for scenario in ['normal', 'low', 'medium', 'high', 'very_high']:
        humidity, desc = get_simulation_scenario(scenario)
        weight, label = get_humidity_weight(humidity)
        print(f"  {scenario}: {humidity:.1f}% - {label}")
