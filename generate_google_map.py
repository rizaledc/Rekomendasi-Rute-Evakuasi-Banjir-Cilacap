import json
import folium
from folium import plugins
import pandas as pd
from pathlib import Path

from src.config import (
    CILACAP_CENTER, MAP_ZOOM_START, MAPS_DIR,
    FLOOD_COLORS, FLOOD_RISK_RADIUS_METERS, MAX_ROUTE_DISTANCE_KM,
    get_humidity_weight, AVERAGE_SPEED_KMH
)
from src.flood_risk import haversine_distance
from src.interactive_map import (
    get_safe_shelters_for_flood_point, 
    get_recommended_shelters,
    precompute_routes,
    calculate_shelter_safety_score
)
from src.data_loader import load_flood_data, load_evacuation_data


def create_google_maps_style(
    flood_df: pd.DataFrame,
    evacuation_df: pd.DataFrame,
    humidity: float = 70.0
) -> folium.Map:
    humidity_weight, humidity_label = get_humidity_weight(humidity)
    
    # Create base map with Google Maps tiles
    m = folium.Map(
        location=CILACAP_CENTER,
        zoom_start=MAP_ZOOM_START,
        tiles=None,  # Don't add default tiles
        control_scale=True
    )
    
    # Add Google Maps tiles (roadmap style - the clean white one)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google Maps',
        name='Google Maps (Peta)',
        max_zoom=20
    ).add_to(m)
    
    # Add satellite option
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satelit',
        max_zoom=20
    ).add_to(m)
    
    # Add hybrid option (satellite + labels)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Google Hybrid',
        max_zoom=20
    ).add_to(m)
    
    # Alternative: OpenStreetMap
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    
    # Prepare flood point data
    print("Preparing flood point data")
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
    
    # Pre-compute routes
    print("Pre-computing routes (this takes a few minutes)")
    route_data = precompute_routes(flood_df, evacuation_df)
    
    # JavaScript with Cancel button
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
    
    function cancelSelection() {{
        // Close all popups
        map.closePopup();
        document.getElementById('route-info').innerHTML = '<p>Pemilihan dibatalkan. Klik titik banjir untuk mencoba lagi.</p>';
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
        var routeColor = '#0066FF';
        if (routeType === 'tercepat') routeColor = '#0066FF';
        else if (routeType === 'teraman') routeColor = '#00AA00';
        else if (routeType === 'seimbang') routeColor = '#FFB300';
        
        currentRoute = L.polyline(latlngs, {{
            color: routeColor,
            weight: 5,
            opacity: 0.8
        }}).addTo(map);
        
        var startMarker = L.marker([flood.lat, flood.lon], {{
            icon: L.divIcon({{
                className: 'start-marker',
                html: '<div style="background: green; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">START</div>',
                iconSize: [60, 30]
            }})
        }}).addTo(map);
        currentMarkers.push(startMarker);
        
        var endMarker = L.marker([shelter.lat, shelter.lon], {{
            icon: L.divIcon({{
                className: 'end-marker',
                html: '<div style="background: ' + routeColor + '; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">TUJUAN</div>',
                iconSize: [70, 30]
            }})
        }}).addTo(map);
        currentMarkers.push(endMarker);
        
        map.fitBounds(currentRoute.getBounds(), {{padding: [50, 50]}});
        
        var typeLabel = routeType ? routeType.toUpperCase() : 'MANUAL';
        var typeEmoji = routeType === 'tercepat' ? '' : (routeType === 'teraman' ? '' : '');
        
        document.getElementById('route-info').innerHTML = 
            '<h4>' + typeEmoji + ' Rute ' + typeLabel + '</h4>' +
            '<p><b>Dari:</b> ' + flood.name + '</p>' +
            '<p><b>Ke:</b> ' + shelter.name + '</p>' +
            '<p><b>Jarak:</b> ' + shelter.distance_km + ' km</p>' +
            '<p><b>Waktu:</b> ' + shelter.travel_time_min + ' menit</p>' +
            '<p><b>Skor Keamanan:</b> ' + (shelter.safety_score || 0).toFixed(1) + '</p>' +
            '<button onclick="clearRoute()" style="background:#ff4444; color:white; padding:5px 10px; border:none; border-radius:5px; cursor:pointer; margin-right:5px;">Hapus Rute</button>';
        
        map.closePopup();
    }}
    
    setTimeout(function() {{
        window.map = document.querySelector('.folium-map')._leaflet_map || 
                     Object.values(window).find(v => v instanceof L.Map);
    }}, 1000);
    </script>
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
        <h4 style="margin:0 0 10px 0;"> Legenda</h4>
        <p style="margin:3px 0;"> Titik Banjir (klik untuk rute)</p>
        <p style="margin:3px 0;"> Shelter Aman</p>
        <p style="margin:3px 0;"> Shelter Kurang Aman</p>
        <hr style="margin:10px 0;">
        <p style="margin:3px 0;"><b>Jenis Rute:</b></p>
        <p style="margin:3px 0;"> Tercepat</p>
        <p style="margin:3px 0;"> Teraman</p>
        <p style="margin:3px 0;"> Seimbang</p>
        <hr style="margin:10px 0;">
        <p style="margin:3px 0; font-size:10px; color:#666;">
            Radius: {FLOOD_RISK_RADIUS_METERS}m | Max: {MAX_ROUTE_DISTANCE_KM}km
        </p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(js_code))
    m.get_root().html.add_child(folium.Element(info_panel))
    m.get_root().html.add_child(folium.Element(legend))
    
    # Add flood markers with Cancel button
    print("Adding flood markers")
    flood_group = folium.FeatureGroup(name='Titik Banjir')
    for idx, fp in flood_df.iterrows():
        flood_id = int(idx)
        flood_info = flood_point_data[flood_id]
        safe_count = len(flood_info['shelters'])
        color = FLOOD_COLORS.get(humidity_weight, '#FF0000')
        
        # Build popup with Cancel button
        popup_content = f'''<div style="width:280px; max-height:350px; overflow-y:auto;">
            <h4 style="margin:0 0 10px 0; color:#FF0000;"> {flood_info['name']}</h4>
            <p style="font-size:10px; color:#666;">Radius: {FLOOD_RISK_RADIUS_METERS}m | Max: {MAX_ROUTE_DISTANCE_KM}km</p>'''
        
        if len(flood_info['shelters']) == 0:
            popup_content += '<p style="color:red;">Tidak ada shelter aman</p>'
        else:
            rec = flood_info['recommendations']
            popup_content += '<div style="background:#f0f8ff; padding:8px; border-radius:5px; margin:5px 0;">'
            popup_content += '<p style="margin:0 0 5px 0;"><b> REKOMENDASI:</b></p>'
            
            for shelter in flood_info['shelters']:
                if rec['tercepat'] == shelter['id']:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'tercepat')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#e3f2fd; border:2px solid #2196f3; border-radius:4px; cursor:pointer; font-size:11px;">
                         <b>Tercepat:</b> {shelter['name'][:25]}<br>
                        <small>{shelter['distance_km']}km • {shelter['travel_time_min']}mnt</small>
                    </button>'''
                    break
            
            for shelter in flood_info['shelters']:
                if rec['teraman'] == shelter['id'] and shelter['id'] != rec['tercepat']:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'teraman')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#e8f5e9; border:2px solid #4caf50; border-radius:4px; cursor:pointer; font-size:11px;">
                         <b>Teraman:</b> {shelter['name'][:25]}<br>
                        <small>{shelter['distance_km']}km • skor:{shelter['safety_score']:.1f}</small>
                    </button>'''
                    break
            
            for shelter in flood_info['shelters']:
                if rec['seimbang'] == shelter['id'] and shelter['id'] != rec['tercepat'] and shelter['id'] != rec['teraman']:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'seimbang')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#fff8e1; border:2px solid #ff9800; border-radius:4px; cursor:pointer; font-size:11px;">
                         <b>Seimbang:</b> {shelter['name'][:25]}<br>
                        <small>{shelter['distance_km']}km • skor:{shelter['safety_score']:.1f}</small>
                    </button>'''
                    break
            
            popup_content += '</div>'
            
            # Other shelters
            other_shelters = [s for s in flood_info['shelters'] 
                           if s['id'] not in [rec['tercepat'], rec['teraman'], rec['seimbang']]]
            
            if other_shelters:
                popup_content += '<p style="margin:10px 0 5px 0;"><b> OPSI LAINNYA:</b></p>'
                popup_content += '<div style="max-height:100px; overflow-y:auto;">'
                for shelter in other_shelters[:6]:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'')" 
                        style="display:block; width:100%; margin:2px 0; padding:4px; text-align:left; 
                        background:#f5f5f5; border:1px solid #ccc; border-radius:4px; cursor:pointer; font-size:10px;">
                         {shelter['name'][:30]}<br>
                        <small style="color:#666;">{shelter['distance_km']}km</small>
                    </button>'''
                popup_content += '</div>'
        
        # Cancel button
        popup_content += '''
        <div style="margin-top:10px; text-align:center;">
            <button onclick="cancelSelection()" 
                style="background:#888; color:white; padding:8px 20px; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">
                 Cancel
            </button>
        </div>
        </div>'''
        
        folium.CircleMarker(
            location=[fp['Latitude'], fp['Longitude']],
            radius=10,
            popup=folium.Popup(popup_content, max_width=320),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f" {fp.get('Desa', 'Titik Banjir')} - {safe_count} shelter"
        ).add_to(flood_group)
    
    flood_group.add_to(m)
    
    # Add evacuation markers
    print("Adding shelter markers")
    evac_group = folium.FeatureGroup(name='Shelter Evakuasi')
    for idx, evac in evacuation_df.iterrows():
        lat, lon = evac['Latitude'], evac['Longitude']
        name = evac.get('Nama_Tempat', evac.get('Nama', f'Evakuasi {idx}'))
        
        is_safe = True
        for _, fp in flood_df.iterrows():
            dist = haversine_distance(lat, lon, fp['Latitude'], fp['Longitude'])
            if dist < FLOOD_RISK_RADIUS_METERS:
                is_safe = False
                break
        
        icon_color = 'green' if is_safe else 'orange'
        
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{name}</b><br>Status: {'Aman' if is_safe else 'Dekat banjir'}",
            icon=folium.Icon(color=icon_color, icon='home', prefix='fa'),
            tooltip=f" {name[:30]}"
        ).add_to(evac_group)
    
    evac_group.add_to(m)
    
    # Heatmap
    heat_data = [[fp['Latitude'], fp['Longitude'], humidity_weight] for _, fp in flood_df.iterrows()]
    plugins.HeatMap(heat_data, name='Heatmap Risiko', min_opacity=0.3, radius=20, blur=15).add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Title
    title = """
    <h3 style="position:fixed; top:20px; left:50%; transform:translateX(-50%); 
               background:white; padding:10px 20px; border-radius:10px; 
               border:2px solid #333; z-index:9999; font-family:Arial;">
         Peta Evakuasi Banjir - Google Maps Style
    </h3>
    """
    m.get_root().html.add_child(folium.Element(title))
    
    return m


if __name__ == "__main__":
    print("Loading data")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    
    print("Creating Google Maps style interactive map")
    m = create_google_maps_style(flood_df, evac_df, humidity=85)
    
    # Save
    output_path = MAPS_DIR / "Peta_Google_Maps.html"
    m.save(str(output_path))
    print(f"\n Saved: {output_path}")
