import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from .config import DATA_DIR
from .flood_risk import haversine_distance


def load_travel_time_lookup() -> Dict[Tuple[str, str], Dict]:
    filepath = DATA_DIR / "Data_Waktu_Tempuh_Final_Fixed.xlsx"
    
    if not filepath.exists():
        print(f"Warning: Travel time data not found at {filepath}")
        return {}
    
    df = pd.read_excel(filepath)
    
    # Normalize column names
    df.columns = [col.strip() for col in df.columns]
    
    lookup = {}
    for _, row in df.iterrows():
        origin = str(row.get('Desa_Asal', '')).strip().lower()
        shelter = str(row.get('Shelter_Tujuan', '')).strip().lower()
        
        if origin and shelter:
            lookup[(origin, shelter)] = {
                'distance_km': float(row.get('Jarak_KM', 0)),
                'time_min': float(row.get('Waktu_Tempuh_Menit', 0)),
                'kondisi': str(row.get('Kondisi', 'NORMAL'))
            }
    
    print(f"Loaded {len(lookup)} travel time records")
    return lookup


def get_travel_info(
    lookup: Dict,
    origin_name: str,
    shelter_name: str
) -> Optional[Dict]:
    # Try exact match first
    key = (origin_name.lower().strip(), shelter_name.lower().strip())
    if key in lookup:
        return lookup[key]
    
    # Try partial matching
    origin_lower = origin_name.lower().strip()
    shelter_lower = shelter_name.lower().strip()
    
    for (o, s), info in lookup.items():
        if origin_lower in o or o in origin_lower:
            if shelter_lower in s or s in shelter_lower:
                return info
    
    return None


def calculate_shelter_safety_score(
    shelter_lat: float,
    shelter_lon: float,
    flood_points: pd.DataFrame
) -> float:
    if len(flood_points) == 0:
        return 100.0  # Max safe
    
    distances = []
    for _, fp in flood_points.iterrows():
        dist = haversine_distance(
            shelter_lat, shelter_lon,
            fp['Latitude'], fp['Longitude']
        ) / 1000  # Convert to km
        distances.append(dist)
    
    # Average distance to all flood points
    avg_distance = sum(distances) / len(distances)
    
    # Minimum distance (closest flood threat)
    min_distance = min(distances)
    
    # Combined score: weighted by min distance (immediate threat) and average
    safety_score = (min_distance * 0.6) + (avg_distance * 0.4)
    
    return round(safety_score, 2)


def rank_shelters_multi_criteria(
    shelters: list,
    flood_points: pd.DataFrame,
    travel_lookup: Dict
) -> Dict[str, Dict]:
    if not shelters:
        return {'tercepat': None, 'teraman': None, 'seimbang': None}
    
    # Add safety scores to shelters
    for shelter in shelters:
        shelter['safety_score'] = calculate_shelter_safety_score(
            shelter['lat'], shelter['lon'], flood_points
        )
    
    # Sort for each criteria
    tercepat = min(shelters, key=lambda s: s['distance_km'])
    teraman = max(shelters, key=lambda s: s['safety_score'])
    
    # Balanced: normalize and combine scores
    max_dist = max(s['distance_km'] for s in shelters) or 1
    max_safety = max(s['safety_score'] for s in shelters) or 1
    
    for shelter in shelters:
        # Lower distance is better (invert), higher safety is better
        dist_score = 1 - (shelter['distance_km'] / max_dist)  # 0-1, higher is better
        safety_norm = shelter['safety_score'] / max_safety  # 0-1, higher is better
        shelter['balanced_score'] = (dist_score * 0.5) + (safety_norm * 0.5)
    
    seimbang = max(shelters, key=lambda s: s['balanced_score'])
    
    # Mark recommendation type
    tercepat['rec_type'] = 'tercepat'
    teraman['rec_type'] = 'teraman'
    seimbang['rec_type'] = 'seimbang'
    
    return {
        'tercepat': tercepat,
        'teraman': teraman,
        'seimbang': seimbang
    }
