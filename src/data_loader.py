"""
Data Loader Module for Flood Evacuation System.
Handles loading and validation of all datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

from .config import (
    FLOOD_DATA_FILE, EVACUATION_DATA_FILE, TRAVEL_TIME_FILE,
    FLOOD_PRONE_VILLAGES_FILE, WEATHER_STATION_FILE, EVACUATION_CSV_FILE,
    CILACAP_BOUNDS
)

warnings.filterwarnings('ignore')


def filter_by_bounds(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Filter dataframe to include only points within Cilacap bounds.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
    
    Returns:
        Filtered DataFrame
    """
    bounds = CILACAP_BOUNDS
    mask = (
        (df[lat_col] >= bounds["south"]) &
        (df[lat_col] <= bounds["north"]) &
        (df[lon_col] >= bounds["west"]) &
        (df[lon_col] <= bounds["east"])
    )
    return df[mask].copy()


def load_flood_data() -> pd.DataFrame:
    """
    Load flood point data from Excel file.
    
    Returns:
        DataFrame with columns: No, Kecamatan, Desa, Latitude, Longitude
    """
    df = pd.read_excel(FLOOD_DATA_FILE)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Find latitude/longitude columns
    lat_cols = [c for c in df.columns if 'lat' in c.lower()]
    lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()]
    
    if lat_cols and lon_cols:
        df = df.rename(columns={lat_cols[0]: 'Latitude', lon_cols[0]: 'Longitude'})
    
    # Convert to numeric
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Drop rows with invalid coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Filter by bounds
    df = filter_by_bounds(df, 'Latitude', 'Longitude')
    
    print(f"Loaded {len(df)} flood points")
    return df


def load_evacuation_data() -> pd.DataFrame:
    """
    Load evacuation point data from Excel file.
    
    Returns:
        DataFrame with evacuation locations
    """
    df = pd.read_excel(EVACUATION_DATA_FILE)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Find latitude/longitude columns
    lat_cols = [c for c in df.columns if 'lat' in c.lower()]
    lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()]
    
    if lat_cols and lon_cols:
        df = df.rename(columns={lat_cols[0]: 'Latitude', lon_cols[0]: 'Longitude'})
    
    # Convert to numeric
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Drop rows with invalid coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Filter by bounds
    df = filter_by_bounds(df, 'Latitude', 'Longitude')
    
    print(f"Loaded {len(df)} evacuation points")
    return df


def load_travel_time_data() -> pd.DataFrame:
    """
    Load travel time matrix data.
    
    Returns:
        DataFrame with travel time/distance between locations
    """
    df = pd.read_excel(TRAVEL_TIME_FILE)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    print(f"Loaded {len(df)} travel time records")
    return df


def load_flood_prone_villages() -> pd.DataFrame:
    """
    Load flood-prone village data.
    
    Returns:
        DataFrame with flood-prone village information
    """
    df = pd.read_excel(FLOOD_PRONE_VILLAGES_FILE)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Find latitude/longitude columns
    lat_cols = [c for c in df.columns if 'lat' in c.lower()]
    lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()]
    
    if lat_cols and lon_cols:
        df = df.rename(columns={lat_cols[0]: 'Latitude', lon_cols[0]: 'Longitude'})
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])
        df = filter_by_bounds(df, 'Latitude', 'Longitude')
    
    print(f"Loaded {len(df)} flood-prone village records")
    return df


def load_weather_data() -> pd.DataFrame:
    """
    Load weather station data (humidity, rainfall).
    
    Returns:
        DataFrame with weather observations
    """
    df = pd.read_excel(WEATHER_STATION_FILE)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Look for humidity column (RH = Relative Humidity)
    rh_cols = [c for c in df.columns if 'rh' in c.lower() or 'humidity' in c.lower() or 'kelembapan' in c.lower()]
    
    # Look for rainfall column (RR = Rainfall)
    rain_cols = [c for c in df.columns if 'rr' in c.lower() or 'rain' in c.lower() or 'hujan' in c.lower()]
    
    if rh_cols:
        df = df.rename(columns={rh_cols[0]: 'Humidity'})
        df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
    
    if rain_cols:
        df = df.rename(columns={rain_cols[0]: 'Rainfall'})
        df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce')
    
    print(f"Loaded {len(df)} weather records")
    return df


def load_all_data() -> dict:
    """
    Load all datasets into a dictionary.
    
    Returns:
        Dictionary with all loaded DataFrames
    """
    return {
        'flood': load_flood_data(),
        'evacuation': load_evacuation_data(),
        'travel_time': load_travel_time_data(),
        'flood_prone': load_flood_prone_villages(),
        'weather': load_weather_data()
    }


if __name__ == "__main__":
    # Test loading all data
    data = load_all_data()
    for name, df in data.items():
        print(f"\n{name}: {len(df)} records")
        print(df.columns.tolist())
