"""
Routing Module using OSMnx.
Handles road network loading and flood-aware route calculation.
"""

import os
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .config import (
    CILACAP_OSM_QUERY, CACHE_DIR, AVERAGE_SPEED_KMH,
    FLOOD_RISK_RADIUS_METERS, FLOOD_WEIGHT_MULTIPLIER
)
from .flood_risk import haversine_distance, calculate_point_flood_risk, get_humidity_weight


# Configure OSMnx
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE_DIR)


def load_road_network(force_download: bool = False) -> nx.MultiDiGraph:
    """
    Load or download Cilacap road network.
    
    Args:
        force_download: If True, download fresh data from OSM
    
    Returns:
        NetworkX MultiDiGraph of the road network
    """
    cache_file = CACHE_DIR / "cilacap_road_network.graphml"
    
    if cache_file.exists() and not force_download:
        print("Loading cached road network...")
        G = ox.load_graphml(cache_file)
    else:
        print("Downloading road network from OpenStreetMap...")
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download network for Cilacap
            G = ox.graph_from_place(
                CILACAP_OSM_QUERY,
                network_type='drive',
                simplify=True
            )
            
            # Save to cache
            ox.save_graphml(G, cache_file)
            print(f"Road network saved to cache: {cache_file}")
        except Exception as e:
            print(f"Error downloading from place query: {e}")
            print("Trying with bounding box...")
            
            # Fallback to bounding box (Cilacap area)
            G = ox.graph_from_bbox(
                north=-7.1, south=-7.9,
                east=109.5, west=108.5,
                network_type='drive',
                simplify=True
            )
            ox.save_graphml(G, cache_file)
    
    print(f"Road network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Find the nearest network node to given coordinates.
    
    Returns:
        Node ID
    """
    return ox.nearest_nodes(G, X=lon, Y=lat)


def calculate_edge_weights(
    G: nx.MultiDiGraph,
    flood_points: pd.DataFrame,
    humidity_weight: int = 1
) -> nx.MultiDiGraph:
    """
    Add flood-aware weights to network edges using batch processing.
    
    Args:
        G: Road network graph
        flood_points: DataFrame with flood point locations
        humidity_weight: Current humidity-based weight (1-5)
    
    Returns:
        Graph with updated edge weights
    """
    print("Calculating flood-aware edge weights...")
    
    # Pre-compute flood point arrays for faster distance calculation
    if len(flood_points) > 0:
        flood_lats = flood_points['Latitude'].values
        flood_lons = flood_points['Longitude'].values
    else:
        flood_lats = np.array([])
        flood_lons = np.array([])
    
    edge_count = 0
    total_edges = len(G.edges())
    
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_count += 1
        
        # Get edge midpoint
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        mid_lat = (u_data['y'] + v_data['y']) / 2
        mid_lon = (u_data['x'] + v_data['x']) / 2
        
        # Get base distance (in meters)
        base_length = data.get('length', 100)
        
        # Calculate minimum distance to any flood point using vectorized operations
        if len(flood_lats) > 0:
            # Simplified distance calculation (faster than haversine for relative comparisons)
            lat_diff = flood_lats - mid_lat
            lon_diff = flood_lons - mid_lon
            distances_sq = lat_diff**2 + lon_diff**2
            min_dist_deg = np.sqrt(np.min(distances_sq))
            
            # Convert to approximate meters (1 degree ~ 111km at equator)
            min_distance = min_dist_deg * 111000
            
            # Calculate risk score
            if min_distance <= FLOOD_RISK_RADIUS_METERS:
                proximity_factor = 1 - (min_distance / FLOOD_RISK_RADIUS_METERS)
                risk_score = proximity_factor * humidity_weight * FLOOD_WEIGHT_MULTIPLIER
            elif min_distance <= FLOOD_RISK_RADIUS_METERS * 2:
                proximity_factor = 1 - ((min_distance - FLOOD_RISK_RADIUS_METERS) / FLOOD_RISK_RADIUS_METERS)
                risk_score = proximity_factor * humidity_weight * 0.5
            else:
                risk_score = 0.1 * humidity_weight
        else:
            risk_score = 0
        
        # Calculate weighted cost
        flood_cost = base_length * (1 + risk_score * FLOOD_WEIGHT_MULTIPLIER)
        
        # Store weights
        G[u][v][key]['flood_weight'] = flood_cost
        G[u][v][key]['flood_risk'] = risk_score
        
        # Progress indicator
        if edge_count % 10000 == 0:
            print(f"  Processed {edge_count}/{total_edges} edges...")
    
    print(f"  Completed processing {total_edges} edges")
    return G


def find_flood_aware_route(
    G: nx.MultiDiGraph,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    flood_points: pd.DataFrame,
    humidity: float = 70.0,
    use_flood_weights: bool = True
) -> Dict:
    """
    Find optimal route considering flood risk.
    
    Args:
        G: Road network graph
        origin: (lat, lon) of starting point
        destination: (lat, lon) of destination
        flood_points: DataFrame with flood point locations
        humidity: Current humidity percentage
        use_flood_weights: If True, avoid flood-prone areas
    
    Returns:
        Dictionary with route information
    """
    # Get humidity weight
    humidity_weight, weight_label = get_humidity_weight(humidity)
    
    print(f"Routing with humidity {humidity}% (weight: {humidity_weight} - {weight_label})")
    
    # Find nearest nodes
    origin_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    # Add flood weights if needed
    if use_flood_weights and len(flood_points) > 0:
        G = calculate_edge_weights(G, flood_points, humidity_weight)
        weight_attr = 'flood_weight'
    else:
        weight_attr = 'length'
    
    try:
        # Find shortest path
        route_nodes = nx.shortest_path(G, origin_node, dest_node, weight=weight_attr)
        
        # Calculate route metrics
        route_coords = []
        total_distance = 0
        total_flood_risk = 0
        
        for i, node in enumerate(route_nodes):
            node_data = G.nodes[node]
            route_coords.append((node_data['y'], node_data['x']))
            
            if i > 0:
                prev_node = route_nodes[i-1]
                edge_data = G.get_edge_data(prev_node, node)
                
                if edge_data:
                    # Get first edge (in case of multiple edges)
                    first_edge = list(edge_data.values())[0]
                    total_distance += first_edge.get('length', 0)
                    total_flood_risk += first_edge.get('flood_risk', 0)
        
        # Calculate travel time (in minutes)
        travel_time_min = (total_distance / 1000) / AVERAGE_SPEED_KMH * 60
        
        return {
            'success': True,
            'route_nodes': route_nodes,
            'route_coords': route_coords,
            'distance_m': total_distance,
            'distance_km': total_distance / 1000,
            'travel_time_min': travel_time_min,
            'total_flood_risk': total_flood_risk,
            'humidity_weight': humidity_weight,
            'humidity_label': weight_label,
            'origin': origin,
            'destination': destination
        }
    
    except nx.NetworkXNoPath:
        return {
            'success': False,
            'error': 'No path found between origin and destination',
            'origin': origin,
            'destination': destination
        }


def find_nearest_safe_evacuation(
    G: nx.MultiDiGraph,
    origin: Tuple[float, float],
    evacuation_points: pd.DataFrame,
    flood_points: pd.DataFrame,
    humidity: float = 70.0,
    max_risk_level: int = 3
) -> Dict:
    """
    Find the nearest safe evacuation point that avoids high flood risk.
    
    Args:
        G: Road network graph
        origin: Starting location (lat, lon)
        evacuation_points: DataFrame with evacuation locations
        flood_points: DataFrame with flood point locations
        humidity: Current humidity percentage
        max_risk_level: Maximum acceptable risk level (1-5)
    
    Returns:
        Route information to best evacuation point
    """
    humidity_weight, _ = get_humidity_weight(humidity)
    
    # Filter evacuation points by safety
    safe_evacuations = []
    
    for idx, evac in evacuation_points.iterrows():
        evac_coord = (evac['Latitude'], evac['Longitude'])
        _, risk_level = calculate_point_flood_risk(evac_coord, flood_points, humidity_weight)
        
        if risk_level <= max_risk_level:
            safe_evacuations.append({
                'idx': idx,
                'coord': evac_coord,
                'name': evac.get('Nama_Tempat', evac.get('Nama', f'Evakuasi {idx}')),
                'risk_level': risk_level
            })
    
    if not safe_evacuations:
        return {
            'success': False,
            'error': f'No evacuation points with risk level <= {max_risk_level}'
        }
    
    print(f"Found {len(safe_evacuations)} safe evacuation points")
    
    # Find shortest route to any safe evacuation
    best_route = None
    best_distance = float('inf')
    
    for evac in safe_evacuations:
        route = find_flood_aware_route(
            G, origin, evac['coord'],
            flood_points, humidity, use_flood_weights=True
        )
        
        if route['success'] and route['distance_m'] < best_distance:
            best_distance = route['distance_m']
            best_route = route
            best_route['evacuation_name'] = evac['name']
            best_route['evacuation_risk'] = evac['risk_level']
    
    if best_route:
        return best_route
    else:
        return {
            'success': False,
            'error': 'Could not find route to any safe evacuation point'
        }


def get_route_geometry(
    G: nx.MultiDiGraph,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    weight: str = 'length'
) -> Optional[List[Tuple[float, float]]]:
    """
    Get road geometry coordinates between two points.
    
    Args:
        G: Road network graph (drive network type)
        origin: (lat, lon) of starting point
        destination: (lat, lon) of destination
        weight: Edge weight to use for shortest path
    
    Returns:
        List of (lat, lon) tuples along the road, or None if no path
    """
    try:
        # Find nearest nodes
        origin_node = find_nearest_node(G, origin[0], origin[1])
        dest_node = find_nearest_node(G, destination[0], destination[1])
        
        # Find shortest path
        route_nodes = nx.shortest_path(G, origin_node, dest_node, weight=weight)
        
        # Extract coordinates
        coords = []
        for node in route_nodes:
            node_data = G.nodes[node]
            coords.append((node_data['y'], node_data['x']))  # (lat, lon)
        
        return coords
    
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def compute_routes_for_flood_point(
    G: nx.MultiDiGraph,
    flood_lat: float,
    flood_lon: float,
    shelters: List[Dict]
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Pre-compute road routes from a flood point to all its shelters.
    
    Args:
        G: Road network graph
        flood_lat: Flood point latitude
        flood_lon: Flood point longitude
        shelters: List of shelter dicts with 'id', 'lat', 'lon'
    
    Returns:
        Dict mapping shelter_id -> list of route coordinates
    """
    routes = {}
    
    for shelter in shelters:
        coords = get_route_geometry(
            G,
            (flood_lat, flood_lon),
            (shelter['lat'], shelter['lon'])
        )
        if coords:
            routes[shelter['id']] = coords
    
    return routes


if __name__ == "__main__":
    # Test the module
    from .data_loader import load_flood_data, load_evacuation_data
    from .config import CILACAP_CENTER
    
    print("Loading data...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    
    print("\nLoading road network...")
    G = load_road_network()
    
    print("\nTesting route calculation...")
    # Route from Alun-alun to first evacuation point
    if len(evac_df) > 0:
        dest = (evac_df.iloc[0]['Latitude'], evac_df.iloc[0]['Longitude'])
        route = find_flood_aware_route(G, CILACAP_CENTER, dest, flood_df, humidity=85)
        
        if route['success']:
            print(f"Route found: {route['distance_km']:.2f} km, {route['travel_time_min']:.1f} min")
        else:
            print(f"Route failed: {route['error']}")
