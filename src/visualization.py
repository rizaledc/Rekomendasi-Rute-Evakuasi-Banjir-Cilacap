import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .config import (
    CILACAP_CENTER, MAP_TILES, MAP_ZOOM_START,
    FLOOD_COLORS, EVACUATION_MARKER_COLOR, FLOOD_MARKER_COLOR,
    ROUTE_COLOR, ROUTE_WEIGHT, MAPS_DIR, get_humidity_weight
)
from .flood_risk import calculate_point_flood_risk


def create_base_map(center: Tuple[float, float] = CILACAP_CENTER) -> folium.Map:
    m = folium.Map(
        location=center,
        zoom_start=MAP_ZOOM_START,
        tiles=MAP_TILES,
        control_scale=True
    )
    
    # Add additional tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('cartodbdark_matter', name='Dark Mode').add_to(m)
    
    return m


def add_flood_markers(
    m: folium.Map,
    flood_points: pd.DataFrame,
    humidity: float = 70.0
) -> folium.Map:
    humidity_weight, weight_label = get_humidity_weight(humidity)
    
    flood_group = folium.FeatureGroup(name=f'Titik Banjir (Risiko: {weight_label})')
    
    for idx, row in flood_points.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        
        # Get location name
        name = row.get('Desa', row.get('Kecamatan', f'Titik Banjir {idx}'))
        kecamatan = row.get('Kecamatan', 'N/A')
        
        # Determine color based on humidity weight
        color = FLOOD_COLORS.get(humidity_weight, FLOOD_MARKER_COLOR)
        
        popup_html = f"""
        <div style="width: 200px;">
            <b>{name}</b><br>
            <i>Kecamatan: {kecamatan}</i><br>
            <hr>
            <b>Tingkat Risiko:</b> {weight_label} ({humidity_weight})<br>
            <b>Koordinat:</b> {lat:.6f}, {lon:.6f}
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            tooltip=f"{name} - Risiko: {weight_label}"
        ).add_to(flood_group)
    
    flood_group.add_to(m)
    return m


def add_evacuation_markers(
    m: folium.Map,
    evacuation_points: pd.DataFrame,
    flood_points: pd.DataFrame = None,
    humidity: float = 70.0
) -> folium.Map:
    evac_group = folium.FeatureGroup(name='Titik Evakuasi')
    
    humidity_weight, _ = get_humidity_weight(humidity)
    
    for idx, row in evacuation_points.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        
        # Get location info
        name = row.get('Nama_Tempat', row.get('Nama', f'Evakuasi {idx}'))
        address = row.get('Alamat', 'N/A')
        floor = row.get('Lantai', row.get('Floor', 'N/A'))
        capacity = row.get('Kapasitas', row.get('Daya_Tampung', 'N/A'))
        
        # Calculate safety level if flood points available
        if flood_points is not None and len(flood_points) > 0:
            _, risk_level = calculate_point_flood_risk((lat, lon), flood_points, humidity_weight)
            safety_status = "Aman" if risk_level <= 2 else ("Sedang" if risk_level <= 3 else "Rawan")
            icon_color = 'green' if risk_level <= 2 else ('orange' if risk_level <= 3 else 'red')
        else:
            safety_status = "N/A"
            icon_color = 'blue'
        
        popup_html = f"""
        <div style="width: 220px;">
            <b style="color: #0066FF;">{name}</b><br>
            <small>{address[:50]}...</small>
            <hr>
            <b>Lantai:</b> {floor}<br>
            <b>Kapasitas:</b> {capacity}<br>
            <b>Status Keamanan:</b> <span style="color: {icon_color};">{safety_status}</span><br>
            <b>Koordinat:</b> {lat:.6f}, {lon:.6f}
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(color=icon_color, icon='home', prefix='fa'),
            tooltip=f"{name} ({safety_status})"
        ).add_to(evac_group)
    
    evac_group.add_to(m)
    return m


def add_flood_heatmap(
    m: folium.Map,
    flood_points: pd.DataFrame,
    humidity: float = 70.0
) -> folium.Map:

    humidity_weight, weight_label = get_humidity_weight(humidity)
    
    # Prepare heatmap data
    heat_data = []
    
    for idx, row in flood_points.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        # Weight by humidity level
        heat_data.append([lat, lon, humidity_weight])
    
    # Add heatmap
    heatmap = plugins.HeatMap(
        heat_data,
        name=f'Heatmap Risiko Banjir ({weight_label})',
        min_opacity=0.3,
        max_zoom=15,
        radius=25,
        blur=15,
        gradient={
            0.2: 'blue',
            0.4: 'lime',
            0.6: 'yellow',
            0.8: 'orange',
            1.0: 'red'
        }
    )
    heatmap.add_to(m)
    
    return m


def add_route(
    m: folium.Map,
    route_info: Dict,
    color: str = ROUTE_COLOR
) -> folium.Map:

    if not route_info.get('success', False):
        return m
    
    route_coords = route_info['route_coords']
    
    # Create route group
    route_group = folium.FeatureGroup(name='Rute Evakuasi')
    
    # Add route line
    folium.PolyLine(
        locations=route_coords,
        weight=ROUTE_WEIGHT,
        color=color,
        opacity=0.8,
        popup=f"Jarak: {route_info['distance_km']:.2f} km<br>Waktu: {route_info['travel_time_min']:.1f} menit"
    ).add_to(route_group)
    
    # Add start marker
    origin = route_info['origin']
    folium.Marker(
        location=origin,
        popup="Titik Awal",
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
        tooltip="Titik Awal"
    ).add_to(route_group)
    
    # Add end marker
    destination = route_info['destination']
    evac_name = route_info.get('evacuation_name', 'Tujuan')
    folium.Marker(
        location=destination,
        popup=f"Tujuan: {evac_name}",
        icon=folium.Icon(color='red', icon='flag', prefix='fa'),
        tooltip=f"Tujuan: {evac_name}"
    ).add_to(route_group)
    
    route_group.add_to(m)
    return m


def add_route_info_box(m: folium.Map, route_info: Dict) -> folium.Map:

    if not route_info.get('success', False):
        return m
    
    info_html = f"""
    <div style="position: fixed; 
                bottom: 10px; 
                left: 10px; 
                width: 280px;
                padding: 10px;
                background-color: white;
                border: 2px solid #333;
                border-radius: 10px;
                z-index: 9999;
                font-family: Arial, sans-serif;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; color: #0066FF;">📍 Informasi Rute</h4>
        <table style="width: 100%; font-size: 12px;">
            <tr>
                <td><b>Jarak:</b></td>
                <td>{route_info['distance_km']:.2f} km</td>
            </tr>
            <tr>
                <td><b>Waktu Tempuh:</b></td>
                <td>{route_info['travel_time_min']:.1f} menit</td>
            </tr>
            <tr>
                <td><b>Tingkat Risiko:</b></td>
                <td>{route_info.get('humidity_label', 'N/A')} ({route_info.get('humidity_weight', 'N/A')})</td>
            </tr>
            <tr>
                <td><b>Tujuan:</b></td>
                <td>{route_info.get('evacuation_name', 'Evakuasi')[:20]}...</td>
            </tr>
        </table>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(info_html))
    return m


def add_legend(m: folium.Map) -> folium.Map:
    legend_html = """
    <div style="position: fixed; 
                top: 10px; 
                right: 10px; 
                width: 180px;
                padding: 10px;
                background-color: white;
                border: 2px solid #333;
                border-radius: 10px;
                z-index: 9999;
                font-family: Arial, sans-serif;
                font-size: 11px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 8px 0;">🗺️ Legenda</h4>
        <p style="margin: 2px 0;"><span style="color: green;">●</span> Evakuasi Aman</p>
        <p style="margin: 2px 0;"><span style="color: orange;">●</span> Evakuasi Sedang</p>
        <p style="margin: 2px 0;"><span style="color: red;">●</span> Evakuasi Rawan</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 2px 0;"><span style="background: #00FF00; padding: 2px 8px;">1</span> Normal</p>
        <p style="margin: 2px 0;"><span style="background: #FFFF00; padding: 2px 8px;">2</span> Rendah</p>
        <p style="margin: 2px 0;"><span style="background: #FFA500; padding: 2px 8px;">3</span> Sedang</p>
        <p style="margin: 2px 0;"><span style="background: #FF4500; padding: 2px 8px; color: white;">4</span> Tinggi</p>
        <p style="margin: 2px 0;"><span style="background: #FF0000; padding: 2px 8px; color: white;">5</span> Sangat Tinggi</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_complete_map(
    flood_points: pd.DataFrame,
    evacuation_points: pd.DataFrame,
    route_info: Optional[Dict] = None,
    humidity: float = 70.0,
    show_heatmap: bool = True,
    title: str = "Sistem Evakuasi Banjir Cilacap"
) -> folium.Map:
    m = create_base_map()
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px; font-family: Arial;">
        <b>{title}</b><br>
        <small>Kelembapan: {humidity}%</small>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add heatmap layer
    if show_heatmap and len(flood_points) > 0:
        m = add_flood_heatmap(m, flood_points, humidity)
    
    # Add flood markers
    if len(flood_points) > 0:
        m = add_flood_markers(m, flood_points, humidity)
    
    # Add evacuation markers
    if len(evacuation_points) > 0:
        m = add_evacuation_markers(m, evacuation_points, flood_points, humidity)
    
    # Add route if available
    if route_info and route_info.get('success', False):
        m = add_route(m, route_info)
        m = add_route_info_box(m, route_info)
    
    # Add legend
    m = add_legend(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m


def save_map(m: folium.Map, filename: str) -> Path:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MAPS_DIR / filename
    m.save(str(filepath))
    print(f"Map saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Test the module
    from .data_loader import load_flood_data, load_evacuation_data
    
    print("Loading data...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    
    print("\nCreating test map...")
    m = create_complete_map(flood_df, evac_df, humidity=85)
    save_map(m, "test_map.html")
