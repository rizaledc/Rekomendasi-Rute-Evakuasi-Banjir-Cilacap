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
    bounds = CILACAP_BOUNDS
    mask = (
        (df[lat_col] >= bounds["south"]) &
        (df[lat_col] <= bounds["north"]) &
        (df[lon_col] >= bounds["west"]) &
        (df[lon_col] <= bounds["east"])
    )
    return df[mask].copy()


def load_flood_data() -> pd.DataFrame:

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
    # Filter by bounds
    df = filter_by_bounds(df, 'Latitude', 'Longitude')
    
    # --- ADD WEATHER DATA FOR MODEL PREDICTION ---
    try:
        weather_df = load_weather_data()
        avg_humidity = weather_df['Humidity'].mean() if 'Humidity' in weather_df else 80.0
        avg_rainfall = weather_df['Rainfall'].mean() if 'Rainfall' in weather_df else 5.0
        
        # Fill missing values with reasonable defaults if weather file is empty/broken
        if pd.isna(avg_humidity): avg_humidity = 80.0
        if pd.isna(avg_rainfall): avg_rainfall = 5.0
        
        print(f"Adding weather context: Humidity={avg_humidity:.1f}%, Rainfall={avg_rainfall:.1f}mm")
        df['Kelembapan'] = avg_humidity
        df['Curah_Hujan'] = avg_rainfall
    except Exception as e:
        print(f"Warning: Could not add weather data: {e}")
        df['Kelembapan'] = 80.0
        df['Curah_Hujan'] = 5.0
        
    print(f"Loaded {len(df)} flood points")
    return df


def load_evacuation_data() -> pd.DataFrame:

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

    df = pd.read_excel(TRAVEL_TIME_FILE)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    print(f"Loaded {len(df)} travel time records")
    return df


def load_flood_prone_villages() -> pd.DataFrame:

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
