"""
Flood Risk Scoring Module.
Calculates flood risk using humidity weights and Random Forest model.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from .config import (
    get_humidity_weight, FLOOD_RISK_RADIUS_METERS,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    FLOOD_WEIGHT_MULTIPLIER
)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in meters.
    """
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def calculate_point_flood_risk(
    point: Tuple[float, float],
    flood_points: pd.DataFrame,
    humidity_weight: int = 1
) -> Tuple[float, int]:
    """
    Calculate flood risk for a point based on proximity to flood areas.
    
    Args:
        point: (latitude, longitude) tuple
        flood_points: DataFrame with flood point locations
        humidity_weight: Current humidity-based weight (1-5)
    
    Returns:
        Tuple of (risk_score, severity_level)
    """
    if len(flood_points) == 0:
        return 0.0, 1
    
    lat, lon = point
    
    # Calculate distances to all flood points
    distances = []
    for _, fp in flood_points.iterrows():
        dist = haversine_distance(lat, lon, fp['Latitude'], fp['Longitude'])
        distances.append(dist)
    
    min_distance = min(distances)
    
    # Calculate risk based on distance
    if min_distance <= FLOOD_RISK_RADIUS_METERS:
        # Inside risk zone - high risk
        proximity_factor = 1 - (min_distance / FLOOD_RISK_RADIUS_METERS)
        risk_score = proximity_factor * humidity_weight * FLOOD_WEIGHT_MULTIPLIER
    elif min_distance <= FLOOD_RISK_RADIUS_METERS * 2:
        # Near risk zone - medium risk
        proximity_factor = 1 - ((min_distance - FLOOD_RISK_RADIUS_METERS) / FLOOD_RISK_RADIUS_METERS)
        risk_score = proximity_factor * humidity_weight * 0.5
    else:
        # Far from risk zone - low risk
        risk_score = 0.1 * humidity_weight
    
    # Determine severity level (1-5)
    if risk_score >= 4:
        severity = 5
    elif risk_score >= 3:
        severity = 4
    elif risk_score >= 2:
        severity = 3
    elif risk_score >= 1:
        severity = 2
    else:
        severity = 1
    
    return risk_score, severity


def calculate_route_flood_risk(
    route_coords: List[Tuple[float, float]],
    flood_points: pd.DataFrame,
    humidity_weight: int = 1
) -> Tuple[float, List[int]]:
    """
    Calculate total flood risk along a route.
    
    Args:
        route_coords: List of (lat, lon) tuples representing the route
        flood_points: DataFrame with flood point locations
        humidity_weight: Current humidity-based weight (1-5)
    
    Returns:
        Tuple of (total_risk, list of severity levels per segment)
    """
    total_risk = 0.0
    severities = []
    
    for coord in route_coords:
        risk, severity = calculate_point_flood_risk(coord, flood_points, humidity_weight)
        total_risk += risk
        severities.append(severity)
    
    return total_risk, severities


class FloodRiskModel:
    """
    Random Forest-based flood risk prediction model.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE
        )
        self.is_trained = False
    
    def prepare_training_data(
        self,
        flood_points: pd.DataFrame,
        evacuation_points: pd.DataFrame,
        weather_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from available datasets.
        
        Features:
        - Distance to nearest flood point
        - Average humidity
        - Number of nearby flood points
        
        Target:
        - Flood risk level (1-5)
        """
        features = []
        labels = []
        
        # Get average humidity for weight calculation
        avg_humidity = weather_data['Humidity'].mean() if 'Humidity' in weather_data.columns else 70
        humidity_weight, _ = get_humidity_weight(avg_humidity)
        
        # Generate training samples from evacuation points
        for _, evac in evacuation_points.iterrows():
            lat, lon = evac['Latitude'], evac['Longitude']
            
            # Calculate features
            distances = []
            for _, fp in flood_points.iterrows():
                dist = haversine_distance(lat, lon, fp['Latitude'], fp['Longitude'])
                distances.append(dist)
            
            min_dist = min(distances) if distances else 10000
            nearby_count = sum(1 for d in distances if d < FLOOD_RISK_RADIUS_METERS * 2)
            
            # Feature vector
            features.append([min_dist, avg_humidity, nearby_count])
            
            # Calculate label (risk level)
            _, severity = calculate_point_flood_risk((lat, lon), flood_points, humidity_weight)
            labels.append(severity)
        
        # Also add flood points as high-risk samples
        for _, fp in flood_points.iterrows():
            features.append([0, avg_humidity, 5])  # At flood point
            labels.append(5)  # High risk
        
        return np.array(features), np.array(labels)
    
    def train(
        self,
        flood_points: pd.DataFrame,
        evacuation_points: pd.DataFrame,
        weather_data: pd.DataFrame
    ) -> dict:
        """
        Train the Random Forest model.
        
        Returns:
            Dictionary with training metrics
        """
        X, y = self.prepare_training_data(flood_points, evacuation_points, weather_data)
        
        if len(X) < 10:
            print("Warning: Not enough data for training. Using rule-based scoring.")
            return {'status': 'insufficient_data'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RF_RANDOM_STATE
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate accuracy
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"Model trained - Train accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")
        
        return {
            'status': 'success',
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': dict(zip(
                ['distance', 'humidity', 'nearby_floods'],
                self.model.feature_importances_
            ))
        }
    
    def predict_risk(
        self,
        lat: float,
        lon: float,
        flood_points: pd.DataFrame,
        humidity: float
    ) -> int:
        """
        Predict flood risk level for a location.
        
        Returns:
            Risk level (1-5)
        """
        if not self.is_trained:
            # Fall back to rule-based scoring
            humidity_weight, _ = get_humidity_weight(humidity)
            _, severity = calculate_point_flood_risk((lat, lon), flood_points, humidity_weight)
            return severity
        
        # Calculate features
        distances = []
        for _, fp in flood_points.iterrows():
            dist = haversine_distance(lat, lon, fp['Latitude'], fp['Longitude'])
            distances.append(dist)
        
        min_dist = min(distances) if distances else 10000
        nearby_count = sum(1 for d in distances if d < FLOOD_RISK_RADIUS_METERS * 2)
        
        # Predict
        features = np.array([[min_dist, humidity, nearby_count]])
        return self.model.predict(features)[0]


if __name__ == "__main__":
    # Test the module
    from .data_loader import load_flood_data, load_evacuation_data, load_weather_data
    
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    weather_df = load_weather_data()
    
    model = FloodRiskModel()
    metrics = model.train(flood_df, evac_df, weather_df)
    print(f"\nModel metrics: {metrics}")
