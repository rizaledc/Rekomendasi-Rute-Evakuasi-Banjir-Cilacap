"""
Interactive Map Generator with Click-to-Route Functionality.
Creates maps where clicking a flood point shows routes to nearby safe evacuation points.
Features: Real road routes, multi-route options (Tercepat/Teraman/Seimbang).
"""

import json
import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import (
    CILACAP_CENTER, MAP_TILES, MAP_ZOOM_START, MAPS_DIR,
    FLOOD_COLORS, FLOOD_RISK_RADIUS_METERS, MAX_ROUTE_DISTANCE_KM,
    get_humidity_weight, AVERAGE_SPEED_KMH
)
from .flood_risk import haversine_distance
from .route_data import rank_shelters_multi_criteria, calculate_shelter_safety_score


def get_safe_shelters_for_flood_point(
    flood_lat: float,
    flood_lon: float,
    evacuation_df: pd.DataFrame,
    flood_df: pd.DataFrame
) -> List[Dict]:
    """
    Get list of safe shelters for a given flood point.
    
    Rules:
    - Shelter must NOT be within 100m of ANY flood point
    - Shelter must be within 5km of the clicked flood point
    """
    safe_shelters = []
    
    for idx, shelter in evacuation_df.iterrows():
        shelter_lat = shelter['Latitude']
        shelter_lon = shelter['Longitude']
        
        # Calculate distance from clicked flood point to shelter
        dist_to_flood = haversine_distance(flood_lat, flood_lon, shelter_lat, shelter_lon)
        dist_km = dist_to_flood / 1000
        
        # Check if within 5km range
        if dist_km > MAX_ROUTE_DISTANCE_KM:
            continue
        
        # Check if shelter is too close to ANY flood point (100m exclusion)
        is_safe = True
        for _, fp in flood_df.iterrows():
            dist_to_any_flood = haversine_distance(
                shelter_lat, shelter_lon, fp['Latitude'], fp['Longitude']
            )
            if dist_to_any_flood < FLOOD_RISK_RADIUS_METERS:
                is_safe = False
                break
        
        if is_safe:
            name = shelter.get('Nama_Tempat', shelter.get('Nama', f'Evakuasi {idx}'))
            travel_time = (dist_km / AVERAGE_SPEED_KMH) * 60  # minutes
            
            # Calculate safety score
            safety_score = calculate_shelter_safety_score(
                shelter_lat, shelter_lon, flood_df
            )
            
            safe_shelters.append({
                'id': int(idx),
                'name': str(name)[:50],
                'lat': float(shelter_lat),
                'lon': float(shelter_lon),
                'distance_km': round(dist_km, 2),
                'travel_time_min': round(travel_time, 1),
                'safety_score': safety_score
            })
    
    # Sort by distance
    safe_shelters.sort(key=lambda x: x['distance_km'])
    
    return safe_shelters


def get_recommended_shelters(shelters: List[Dict]) -> Dict[str, Optional[Dict]]:
    """Get the 3 recommended shelters: tercepat, teraman, seimbang."""
    if not shelters:
        return {'tercepat': None, 'teraman': None, 'seimbang': None}
    
    # Tercepat: shortest distance
    tercepat = min(shelters, key=lambda s: s['distance_km'])
    
    # Teraman: highest safety score
    teraman = max(shelters, key=lambda s: s['safety_score'])
    
    # Seimbang: balanced score
    max_dist = max(s['distance_km'] for s in shelters) or 1
    max_safety = max(s['safety_score'] for s in shelters) or 1
    
    for shelter in shelters:
        dist_score = 1 - (shelter['distance_km'] / max_dist)
        safety_norm = shelter['safety_score'] / max_safety
        shelter['balanced_score'] = (dist_score * 0.5) + (safety_norm * 0.5)
    
    seimbang = max(shelters, key=lambda s: s.get('balanced_score', 0))
    
    return {
        'tercepat': tercepat,
        'teraman': teraman,
        'seimbang': seimbang
    }


def precompute_routes(
    flood_df: pd.DataFrame,
    evacuation_df: pd.DataFrame,
    road_graph=None
) -> Dict:
    """
    Pre-compute all route geometries from flood points to shelters.
    Returns dict: flood_id -> shelter_id -> list of (lat, lon) coordinates
    """
    from .routing import load_road_network, get_route_geometry
    
    print("Pre-computing road routes...")
    
    # Load road network if not provided
    if road_graph is None:
        road_graph = load_road_network()
    
    all_routes = {}
    total_flood_points = len(flood_df)
    
    for flood_idx, fp in flood_df.iterrows():
        flood_id = int(flood_idx)
        all_routes[flood_id] = {}
        
        # Get safe shelters for this flood point
        shelters = get_safe_shelters_for_flood_point(
            fp['Latitude'], fp['Longitude'],
            evacuation_df, flood_df
        )
        
        # Compute routes to each shelter
        for shelter in shelters[:15]:  # Limit to top 15 for performance
            route_coords = get_route_geometry(
                road_graph,
                (fp['Latitude'], fp['Longitude']),
                (shelter['lat'], shelter['lon'])
            )
            if route_coords:
                all_routes[flood_id][shelter['id']] = route_coords
        
        if (flood_idx + 1) % 20 == 0:
            print(f"  Processed {flood_idx + 1}/{total_flood_points} flood points...")
    
    print(f"Pre-computed routes for {len(all_routes)} flood points")
    return all_routes


def create_interactive_flood_map(
    flood_df: pd.DataFrame,
    evacuation_df: pd.DataFrame,
    humidity: float = 70.0,
    precompute: bool = True
) -> folium.Map:
    """
    Create interactive map with real road routes and multi-route options.
    """
    humidity_weight, humidity_label = get_humidity_weight(humidity)
    
    # Create base map
    m = folium.Map(
        location=CILACAP_CENTER,
        zoom_start=MAP_ZOOM_START,
        tiles=MAP_TILES,
        control_scale=True
    )
    
    # Add tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('cartodbdark_matter', name='Dark Mode').add_to(m)
    
    # Prepare flood point data with shelters and recommendations
    flood_point_data = {}
    for idx, fp in flood_df.iterrows():
        shelters = get_safe_shelters_for_flood_point(
            fp['Latitude'], fp['Longitude'],
            evacuation_df, flood_df
        )
        recommendations = get_recommended_shelters(shelters)
        
        flood_point_data[int(idx)] = {
            'name': str(fp.get('Desa', fp.get('Kecamatan', f'Titik Banjir {idx}')))[:30],
            'lat': float(fp['Latitude']),
            'lon': float(fp['Longitude']),
            'shelters': shelters,
            'recommendations': {
                'tercepat': recommendations['tercepat']['id'] if recommendations['tercepat'] else None,
                'teraman': recommendations['teraman']['id'] if recommendations['teraman'] else None,
                'seimbang': recommendations['seimbang']['id'] if recommendations['seimbang'] else None
            }
        }
    
    # Pre-compute routes if enabled
    route_data = {}
    if precompute:
        route_data = precompute_routes(flood_df, evacuation_df)
    
    # Add JavaScript for interactivity
    js_code = f"""
    <script>
    var floodData = {json.dumps(flood_point_data)};
    var routeData = {json.dumps(route_data)};
    var currentRoute = null;
    var currentMarkers = [];
    
    function clearRoute() {{
        if (currentRoute) {{
            map.removeLayer(currentRoute);
            currentRoute = null;
        }}
        currentMarkers.forEach(function(m) {{
            map.removeLayer(m);
        }});
        currentMarkers = [];
        document.getElementById('route-info').innerHTML = '<p>Klik titik banjir (merah) untuk melihat shelter tersedia</p>';
    }}
    
    function showRoute(floodId, shelterId, routeType) {{
        clearRoute();
        
        var flood = floodData[floodId];
        var shelter = flood.shelters.find(s => s.id === shelterId);
        
        if (!shelter) return;
        
        // Get pre-computed route or fallback to straight line
        var latlngs;
        if (routeData[floodId] && routeData[floodId][shelterId]) {{
            latlngs = routeData[floodId][shelterId];
        }} else {{
            latlngs = [[flood.lat, flood.lon], [shelter.lat, shelter.lon]];
        }}
        
        // Route color based on type
        var routeColor = '#0066FF';  // Default blue
        if (routeType === 'tercepat') routeColor = '#0066FF';
        else if (routeType === 'teraman') routeColor = '#00AA00';
        else if (routeType === 'seimbang') routeColor = '#FFB300';
        
        currentRoute = L.polyline(latlngs, {{
            color: routeColor,
            weight: 5,
            opacity: 0.8
        }}).addTo(map);
        
        // Add start marker
        var startMarker = L.marker([flood.lat, flood.lon], {{
            icon: L.divIcon({{
                className: 'start-marker',
                html: '<div style="background: green; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">START</div>',
                iconSize: [60, 30]
            }})
        }}).addTo(map);
        currentMarkers.push(startMarker);
        
        // Add end marker  
        var endMarker = L.marker([shelter.lat, shelter.lon], {{
            icon: L.divIcon({{
                className: 'end-marker',
                html: '<div style="background: ' + routeColor + '; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">TUJUAN</div>',
                iconSize: [70, 30]
            }})
        }}).addTo(map);
        currentMarkers.push(endMarker);
        
        // Fit bounds
        map.fitBounds(currentRoute.getBounds(), {{padding: [50, 50]}});
        
        // Route type label
        var typeLabel = routeType ? routeType.toUpperCase() : 'MANUAL';
        var typeEmoji = routeType === 'tercepat' ? '🔵' : (routeType === 'teraman' ? '🟢' : '🟡');
        
        // Update info
        document.getElementById('route-info').innerHTML = 
            '<h4>' + typeEmoji + ' Rute ' + typeLabel + '</h4>' +
            '<p><b>Dari:</b> ' + flood.name + '</p>' +
            '<p><b>Ke:</b> ' + shelter.name + '</p>' +
            '<p><b>Jarak:</b> ' + shelter.distance_km + ' km</p>' +
            '<p><b>Waktu:</b> ' + shelter.travel_time_min + ' menit</p>' +
            '<p><b>Skor Keamanan:</b> ' + (shelter.safety_score || 0).toFixed(1) + '</p>' +
            '<button onclick="clearRoute()" style="background:#ff4444; color:white; padding:5px 10px; border:none; border-radius:5px; cursor:pointer;">Hapus Rute</button>';
    }}
    
    // Wait for map to load
    setTimeout(function() {{
        window.map = document.querySelector('.folium-map')._leaflet_map || 
                     Object.values(window).find(v => v instanceof L.Map);
    }}, 1000);
    </script>
    """
    
    # Add route info panel
    info_panel = """
    <div id="route-info" style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        width: 280px;
        padding: 15px;
        background: white;
        border: 2px solid #333;
        border-radius: 10px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        font-size: 13px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <p>Klik titik banjir (merah) untuk melihat shelter tersedia</p>
    </div>
    """
    
    # Add legend
    legend = f"""
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        width: 220px;
        padding: 15px;
        background: white;
        border: 2px solid #333;
        border-radius: 10px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        font-size: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <h4 style="margin:0 0 10px 0;">🗺️ Legenda</h4>
        <p style="margin:3px 0;">🔴 Titik Banjir (klik untuk rute)</p>
        <p style="margin:3px 0;">🟢 Shelter Aman</p>
        <p style="margin:3px 0;">🟡 Shelter Kurang Aman</p>
        <hr style="margin:10px 0;">
        <p style="margin:3px 0;"><b>Jenis Rute:</b></p>
        <p style="margin:3px 0;">🔵 Tercepat (jarak terpendek)</p>
        <p style="margin:3px 0;">🟢 Teraman (risiko terendah)</p>
        <p style="margin:3px 0;">🟡 Seimbang (balanced)</p>
        <hr style="margin:10px 0;">
        <p style="margin:3px 0; font-size:10px; color:#666;">
            Radius exclusion: {FLOOD_RISK_RADIUS_METERS}m<br>
            Max jarak: {MAX_ROUTE_DISTANCE_KM}km<br>
            Kelembapan: {humidity:.1f}% ({humidity_label})
        </p>
    </div>
    """
    
    # Add HTML elements
    m.get_root().html.add_child(folium.Element(js_code))
    m.get_root().html.add_child(folium.Element(info_panel))
    m.get_root().html.add_child(folium.Element(legend))
    
    # Add flood markers with multi-route popup
    flood_group = folium.FeatureGroup(name='Titik Banjir (Klik untuk rute)')
    for idx, fp in flood_df.iterrows():
        flood_id = int(idx)
        flood_info = flood_point_data[flood_id]
        safe_count = len(flood_info['shelters'])
        color = FLOOD_COLORS.get(humidity_weight, '#FF0000')
        
        # Build popup content with recommendations + all shelters
        popup_content = f'''<div style="width:280px; max-height:350px; overflow-y:auto;">
            <h4 style="margin:0 0 10px 0; color:#FF0000;">🌊 {flood_info['name']}</h4>
            <p style="font-size:10px; color:#666;">Radius: {FLOOD_RISK_RADIUS_METERS}m | Max: {MAX_ROUTE_DISTANCE_KM}km</p>'''
        
        if len(flood_info['shelters']) == 0:
            popup_content += '<p style="color:red;">Tidak ada shelter aman</p>'
        else:
            # Recommendations section
            rec = flood_info['recommendations']
            popup_content += '<div style="background:#f0f8ff; padding:8px; border-radius:5px; margin:5px 0;">'
            popup_content += '<p style="margin:0 0 5px 0;"><b>🎯 REKOMENDASI:</b></p>'
            
            for shelter in flood_info['shelters']:
                if rec['tercepat'] == shelter['id']:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'tercepat')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#e3f2fd; border:2px solid #2196f3; border-radius:4px; cursor:pointer; font-size:11px;">
                        🔵 <b>Tercepat:</b> {shelter['name'][:25]}<br>
                        <small>{shelter['distance_km']}km • {shelter['travel_time_min']}mnt</small>
                    </button>'''
                    break
            
            for shelter in flood_info['shelters']:
                if rec['teraman'] == shelter['id'] and shelter['id'] != rec['tercepat']:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'teraman')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#e8f5e9; border:2px solid #4caf50; border-radius:4px; cursor:pointer; font-size:11px;">
                        🟢 <b>Teraman:</b> {shelter['name'][:25]}<br>
                        <small>{shelter['distance_km']}km • skor:{shelter['safety_score']:.1f}</small>
                    </button>'''
                    break
            
            for shelter in flood_info['shelters']:
                if rec['seimbang'] == shelter['id'] and shelter['id'] != rec['tercepat'] and shelter['id'] != rec['teraman']:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'seimbang')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#fff8e1; border:2px solid #ff9800; border-radius:4px; cursor:pointer; font-size:11px;">
                        🟡 <b>Seimbang:</b> {shelter['name'][:25]}<br>
                        <small>{shelter['distance_km']}km • skor:{shelter['safety_score']:.1f}</small>
                    </button>'''
                    break
            
            popup_content += '</div>'
            
            # Other shelters section
            other_shelters = [s for s in flood_info['shelters'] 
                           if s['id'] not in [rec['tercepat'], rec['teraman'], rec['seimbang']]]
            
            if other_shelters:
                popup_content += '<p style="margin:10px 0 5px 0;"><b>📋 OPSI LAINNYA:</b></p>'
                popup_content += '<div style="max-height:120px; overflow-y:auto;">'
                for shelter in other_shelters[:8]:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'')" 
                        style="display:block; width:100%; margin:2px 0; padding:4px; text-align:left; 
                        background:#f5f5f5; border:1px solid #ccc; border-radius:4px; cursor:pointer; font-size:10px;">
                        🏠 {shelter['name'][:30]}<br>
                        <small style="color:#666;">{shelter['distance_km']}km • {shelter['travel_time_min']}mnt</small>
                    </button>'''
                if len(other_shelters) > 8:
                    popup_content += f'<p style="font-size:9px; color:#666;">+{len(other_shelters)-8} lainnya</p>'
                popup_content += '</div>'
        
        popup_content += '</div>'
        
        folium.CircleMarker(
            location=[fp['Latitude'], fp['Longitude']],
            radius=10,
            popup=folium.Popup(popup_content, max_width=320),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f"🌊 {fp.get('Desa', 'Titik Banjir')} - {safe_count} shelter"
        ).add_to(flood_group)
    
    flood_group.add_to(m)
    
    # Add evacuation markers
    evac_group = folium.FeatureGroup(name='Titik Evakuasi')
    for idx, evac in evacuation_df.iterrows():
        lat, lon = evac['Latitude'], evac['Longitude']
        name = evac.get('Nama_Tempat', evac.get('Nama', f'Evakuasi {idx}'))
        
        # Check if safe
        is_safe = True
        for _, fp in flood_df.iterrows():
            dist = haversine_distance(lat, lon, fp['Latitude'], fp['Longitude'])
            if dist < FLOOD_RISK_RADIUS_METERS:
                is_safe = False
                break
        
        icon_color = 'green' if is_safe else 'orange'
        
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{name}</b><br>Status: {'Aman' if is_safe else 'Terlalu dekat banjir'}",
            icon=folium.Icon(color=icon_color, icon='home', prefix='fa'),
            tooltip=f"🏠 {name[:30]}"
        ).add_to(evac_group)
    
    evac_group.add_to(m)
    
    # Add heatmap
    heat_data = [[fp['Latitude'], fp['Longitude'], humidity_weight] for _, fp in flood_df.iterrows()]
    plugins.HeatMap(
        heat_data,
        name='Heatmap Risiko',
        min_opacity=0.3,
        radius=20,
        blur=15
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add title
    title = """
    <h3 style="position:fixed; top:20px; left:50%; transform:translateX(-50%); 
               background:white; padding:10px 20px; border-radius:10px; 
               border:2px solid #333; z-index:9999; font-family:Arial;">
        Sistem Evakuasi Banjir Interaktif - Cilacap
    </h3>
    """
    m.get_root().html.add_child(folium.Element(title))
    
    return m


def save_interactive_map(m: folium.Map, filename: str = "peta_interaktif.html") -> Path:
    """Save the interactive map."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MAPS_DIR / filename
    m.save(str(filepath))
    print(f"Interactive map saved: {filepath}")
    return filepath


if __name__ == "__main__":
    from .data_loader import load_flood_data, load_evacuation_data
    
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    
    m = create_interactive_flood_map(flood_df, evac_df, humidity=85)
    save_interactive_map(m)
