"""
Configuration module for Flood Evacuation Route Recommendation System.
Contains all constants, file paths, and configuration settings.
"""

import os
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Banjir Cilacap"
OUTPUT_DIR = PROJECT_ROOT / "output"
MAPS_DIR = OUTPUT_DIR / "maps"
REPORTS_DIR = OUTPUT_DIR / "reports"
CACHE_DIR = PROJECT_ROOT / "cache"

# =============================================================================
# Data Files
# =============================================================================
FLOOD_DATA_FILE = DATA_DIR / "Data_Banjir_FIX_120.xlsx"
EVACUATION_DATA_FILE = DATA_DIR / "Data_Evakuasi_FIX_314.xlsx"
TRAVEL_TIME_FILE = DATA_DIR / "Data_Waktu_Tempuh_Final_Fixed.xlsx"
FLOOD_PRONE_VILLAGES_FILE = DATA_DIR / "Data Desa Rawan Banjr di Cilacap.xlsx"
WEATHER_STATION_FILE = DATA_DIR / "Stasiun Metalurgi Tunggul Wulung.xlsx"
EVACUATION_CSV_FILE = DATA_DIR / "tempatevakuasinew.csv"

# =============================================================================
# Geographic Configuration - Cilacap
# =============================================================================
CILACAP_CENTER = (-7.727456, 109.009519)  # Alun-alun Cilacap
CILACAP_BOUNDS = {
    "north": -7.1,
    "south": -7.9,
    "east": 109.5,
    "west": 108.5
}

# OpenStreetMap query location
CILACAP_OSM_QUERY = "Cilacap, Central Java, Indonesia"

# =============================================================================
# Humidity to Flood Weight Mapping (BMKG Standard)
# =============================================================================
HUMIDITY_WEIGHTS = {
    "normal": {"range": (0, 70), "weight": 1, "label": "Normal"},
    "low": {"range": (71, 80), "weight": 2, "label": "Rendah"},
    "medium": {"range": (81, 90), "weight": 3, "label": "Sedang"},
    "high": {"range": (91, 95), "weight": 4, "label": "Tinggi"},
    "very_high": {"range": (96, 100), "weight": 5, "label": "Sangat Tinggi"}
}

def get_humidity_weight(humidity: float) -> tuple:
    """
    Get flood weight based on humidity percentage.
    
    Args:
        humidity: Humidity percentage (0-100)
    
    Returns:
        tuple: (weight, label)
    """
    if humidity <= 70:
        return 1, "Normal"
    elif humidity <= 80:
        return 2, "Rendah"
    elif humidity <= 90:
        return 3, "Sedang"
    elif humidity <= 95:
        return 4, "Tinggi"
    else:
        return 5, "Sangat Tinggi"

# =============================================================================
# Routing Configuration
# =============================================================================
AVERAGE_SPEED_KMH = 30  # Average speed for travel time calculation
FLOOD_RISK_RADIUS_METERS = 100  # Radius around flood points - shelters within this cannot be selected
MAX_ROUTE_DISTANCE_KM = 5  # Maximum route distance from flood point to shelter
FLOOD_WEIGHT_MULTIPLIER = 2.0  # Weight multiplier for flood-prone areas

# =============================================================================
# Visualization Configuration
# =============================================================================
MAP_TILES = "CartoDB positron"
MAP_ZOOM_START = 11

# Color scheme for flood severity
FLOOD_COLORS = {
    1: "#00FF00",  # Green - Normal
    2: "#FFFF00",  # Yellow - Low
    3: "#FFA500",  # Orange - Medium
    4: "#FF4500",  # Red-Orange - High
    5: "#FF0000"   # Red - Very High
}

EVACUATION_MARKER_COLOR = "blue"
FLOOD_MARKER_COLOR = "red"
ROUTE_COLOR = "#0066FF"
ROUTE_WEIGHT = 4

# =============================================================================
# Random Forest Configuration
# =============================================================================
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
