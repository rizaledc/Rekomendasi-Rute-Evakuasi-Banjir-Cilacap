import ee
import json
import folium
from folium import plugins
import pandas as pd
import osmnx as ox
from pathlib import Path

from src.config import (
    CILACAP_BOUNDS, CILACAP_CENTER, MAP_ZOOM_START, MAPS_DIR,
    FLOOD_COLORS, FLOOD_RISK_RADIUS_METERS, MAX_ROUTE_DISTANCE_KM,
    get_humidity_weight, AVERAGE_SPEED_KMH
)
from src.flood_risk import haversine_distance
from src.interactive_map import (
    get_safe_shelters_for_flood_point, 
    get_recommended_shelters,
    precompute_routes
)
from src.data_loader import load_flood_data, load_evacuation_data
from src.satellite_analysis import GEE_PROJECT_ID


DATE_START = '2024-10-01'
DATE_END = '2024-12-16'


def initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"GEE initialized with project: {GEE_PROJECT_ID}")
        return True
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        return False


def get_cilacap_polygon_from_osm():
    try:
        gdf = ox.geocode_to_gdf("Kabupaten Cilacap, Jawa Tengah, Indonesia")
        geometry = gdf.geometry.iloc[0]
        
        # Extract polygon coordinates
        if geometry.geom_type == 'Polygon':
            coords = list(geometry.exterior.coords)
        elif geometry.geom_type == 'MultiPolygon':
            # Use the largest polygon
            largest = max(geometry.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
        else:
            raise ValueError(f"Unexpected geometry type: {geometry.geom_type}")
        
        # Convert to [lon, lat] format for EE
        ee_coords = [[lon, lat] for lon, lat in coords]
        
        print(f"Got Cilacap polygon with {len(coords)} points")
        return ee_coords, coords
        
    except Exception as e:
        print(f"Warning: Could not fetch from OSM: {e}")
        print("Using approximate Cilacap bounds")
        # Fallback to approximate bounds following coastline
        approx_coords = [
            (108.80, -7.75),
            (108.90, -7.70),
            (109.00, -7.55),
            (109.15, -7.50),
            (109.30, -7.55),
            (109.45, -7.60),
            (109.45, -7.80),
            (109.30, -7.75),
            (109.15, -7.78),
            (109.00, -7.80),
            (108.90, -7.82),
            (108.80, -7.80),
            (108.80, -7.75)
        ]
        ee_coords = [[lon, lat] for lon, lat in approx_coords]
        folium_coords = [(lat, lon) for lon, lat in approx_coords]
        return ee_coords, approx_coords


def get_satellite_layers(cilacap_polygon_ee):
    
    geometry = ee.Geometry.Polygon([cilacap_polygon_ee])
    
    # Get Sentinel-2 composite
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    count = collection.size().getInfo()
    print(f"Found {count} Sentinel-2 images")
    
    if count == 0:
        raise ValueError("No satellite images found for the date range")
    
    composite = collection.median().clip(geometry)
    
    # Calculate NDWI: (Green - NIR) / (Green + NIR)
    green = composite.select('B3')
    nir = composite.select('B8')
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    # Calculate NDMI: (NIR - SWIR) / (NIR + SWIR)
    swir = composite.select('B11')
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename('NDMI')
    
    # Water mask (NDWI > 0.3)
    water_mask = ndwi.gt(0.3).selfMask()
    
    # Flood risk classification from satellite
    risk_class = ee.Image(0).where(ndwi.gt(-0.2), 1)
    risk_class = risk_class.where(ndwi.gt(0), 2)
    risk_class = risk_class.where(ndwi.gt(0.2), 3)
    risk_class = risk_class.where(ndwi.gt(0.4), 4)
    risk_class = risk_class.clip(geometry)
    
    # Get tile URLs
    ndwi_url = ndwi.getMapId({
        'min': -0.5, 'max': 0.5,
        'palette': ['brown', 'yellow', 'lightgreen', 'cyan', 'blue']
    })['tile_fetcher'].url_format
    
    ndmi_url = ndmi.getMapId({
        'min': -0.3, 'max': 0.5,
        'palette': ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
    })['tile_fetcher'].url_format
    
    water_url = water_mask.getMapId({
        'palette': ['0066FF']
    })['tile_fetcher'].url_format
    
    risk_url = risk_class.getMapId({
        'min': 0, 'max': 4,
        'palette': ['green', 'yellow', 'orange', 'red', 'darkred']
    })['tile_fetcher'].url_format
    
    return {
        'ndwi': ndwi_url,
        'ndmi': ndmi_url,
        'water': water_url,
        'risk': risk_url
    }


def create_integrated_map():
    print("PETA TERINTEGRASI: SATELIT + EVAKUASI BANJIR")
    print("=" * 60)
    
    # Initialize GEE
    if not initialize_gee():
        raise RuntimeError("Failed to initialize GEE")
    
    # Get Cilacap polygon
    print("\nGetting Cilacap boundary")
    cilacap_ee, cilacap_coords = get_cilacap_polygon_from_osm()
    
    # Convert for Folium (lat, lon format)
    folium_polygon = [[lat, lon] for lon, lat in cilacap_coords] if isinstance(cilacap_coords[0], tuple) else [[c[1], c[0]] for c in cilacap_coords]
    
    # Load data
    print("\nLoading flood and evacuation data")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    print(f"Flood points: {len(flood_df)}, Shelters: {len(evac_df)}")
    
    # Get satellite layers clipped to polygon
    print("\nProcessing satellite imagery (clipped to Cilacap polygon)")
    sat_layers = get_satellite_layers(cilacap_ee)
    
    # Create base map
    print("\nCreating base map")
    m = folium.Map(
        location=CILACAP_CENTER,
        zoom_start=10,
        tiles=None
    )
    
    # Add Google Maps tiles
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google Maps',
        name='Google Maps (Peta)',
        max_zoom=20
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satelit',
        max_zoom=20
    ).add_to(m)
    
    # Add Cilacap boundary polygon
    folium.Polygon(
        locations=folium_polygon,
        color='#0066FF',
        weight=3,
        fill=False,
        popup='Batas Kabupaten Cilacap',
        tooltip='Batas Wilayah Cilacap'
    ).add_to(m)
    
    # Add satellite layers (clipped to polygon)
    print("\nAdding satellite layers")
    
    folium.TileLayer(
        tiles=sat_layers['ndwi'],
        attr='NDWI',
        name=' NDWI (Water Index)',
        overlay=True,
        show=False,
        opacity=0.7
    ).add_to(m)
    
    folium.TileLayer(
        tiles=sat_layers['ndmi'],
        attr='NDMI',
        name=' NDMI (Moisture)',
        overlay=True,
        show=False,
        opacity=0.7
    ).add_to(m)
    
    folium.TileLayer(
        tiles=sat_layers['water'],
        attr='Water',
        name=' Badan Air (NDWI>0.3)',
        overlay=True,
        show=False,
        opacity=0.6
    ).add_to(m)
    
    folium.TileLayer(
        tiles=sat_layers['risk'],
        attr='Risk',
        name=' Klasifikasi Risiko Banjir',
        overlay=True,
        show=True,
        opacity=0.5
    ).add_to(m)
    
    # Prepare flood point data with shelters
    print("\nAdding evacuation data with routes")
    flood_point_data = {}
    for idx, fp in flood_df.iterrows():
        # Get safe shelters (100m exclusion, 5km max distance)
        shelters = get_safe_shelters_for_flood_point(
            fp['Latitude'], fp['Longitude'],
            evac_df, flood_df
        )
        # Get 3 recommendations
        recommendations = get_recommended_shelters(shelters, fp)
        
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
    
    # Pre-compute routes using OSMnx drive network
    print(f"Pre-computing road routes (OSMnx drive network)")
    route_data = precompute_routes(flood_df, evac_df)
    
    # JavaScript for route handling
    js_code = f"""
    <script>
    var floodData = {json.dumps(flood_point_data)};
    var routeData = {json.dumps(route_data)};
    var currentRoute = null;
    var currentMarkers = [];
    
    function clearRoute() {{
        if (currentRoute) {{ map.removeLayer(currentRoute); currentRoute = null; }}
        currentMarkers.forEach(function(m) {{ map.removeLayer(m); }});
        currentMarkers = [];
        document.getElementById('route-info').innerHTML = '<p>Klik titik banjir untuk melihat rute</p>';
    }}
    
    function showRoute(floodId, shelterId, routeType) {{
        clearRoute();
        var flood = floodData[floodId];
        var shelter = flood.shelters.find(s => s.id === shelterId);
        if (!shelter) return;
        
        // Get route data
        var routeInfo = routeData[floodId] && routeData[floodId][shelterId];
        
        var routeColor = routeType === 'tercepat' ? '#0066FF' 
            : (routeType === 'teraman' ? '#00AA00' : '#FFB300');
        var seaColor = '#87CEEB';  // Light blue for ferry crossing
        
        var allBounds = [];
        
        // Check if cross-sea route (has segments property)
        if (routeInfo && routeInfo.segments) {{
            // Multi-segment cross-sea route
            routeInfo.segments.forEach(function(seg) {{
                var color = seg.type === 'sea' ? seaColor : routeColor;
                var weight = seg.type === 'sea' ? 4 : 6;
                var dashArray = seg.type === 'sea' ? '10, 10' : null;
                
                var polyline = L.polyline(seg.coords, {{
                    color: color, 
                    weight: weight, 
                    opacity: 0.9,
                    dashArray: dashArray
                }}).addTo(map);
                currentMarkers.push(polyline);
                allBounds.push(polyline.getBounds());
            }});
        }} else {{
            // Simple route
            var latlngs = routeInfo ? routeInfo : [[flood.lat, flood.lon], [shelter.lat, shelter.lon]];
            currentRoute = L.polyline(latlngs, {{color: routeColor, weight: 6, opacity: 0.9}}).addTo(map);
            allBounds.push(currentRoute.getBounds());
        }}
        
        // Add markers
        var startMarker = L.marker([flood.lat, flood.lon], {{
            icon: L.divIcon({{html: '<div style="background:green;color:white;padding:3px 8px;border-radius:4px;font-weight:bold;font-size:11px;">START</div>', iconSize:[50,25]}})
        }}).addTo(map);
        currentMarkers.push(startMarker);
        
        var endMarker = L.marker([shelter.lat, shelter.lon], {{
            icon: L.divIcon({{html: '<div style="background:'+routeColor+';color:white;padding:3px 8px;border-radius:4px;font-weight:bold;font-size:11px;">TUJUAN</div>', iconSize:[55,25]}})
        }}).addTo(map);
        currentMarkers.push(endMarker);
        
        // Fit bounds to all segments
        if (allBounds.length > 0) {{
            var combinedBounds = allBounds[0];
            allBounds.forEach(function(b) {{ combinedBounds.extend(b); }});
            map.fitBounds(combinedBounds, {{padding: [40, 40]}});
        }}
        
        // Update info panel
        var typeLabel = routeType === 'tercepat' ? ' TERCEPAT' : (routeType === 'teraman' ? ' TERAMAN' : ' SEIMBANG');
        document.getElementById('route-info').innerHTML = 
            '<h4>' + typeLabel + '</h4>' +
            '<p><b>Dari:</b> ' + flood.name + '</p>' +
            '<p><b>Ke:</b> ' + shelter.name + '</p>' +
            '<p><b>Jarak:</b> ' + shelter.distance_km + ' km</p>' +
            '<p><b>Waktu:</b> ' + shelter.travel_time_min + ' menit</p>' +
            '<button onclick="clearRoute()" style="background:#ff4444;color:white;padding:5px 10px;border:none;border-radius:4px;cursor:pointer;">Hapus Rute</button>';
        
        map.closePopup();
    }}
    
    function cancelSelection() {{ map.closePopup(); }}
    
    setTimeout(function() {{
        window.map = Object.values(window).find(v => v instanceof L.Map);
    }}, 1000);
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_code))
    
    # Info panel
    info_panel = """
    <div id="route-info" style="position:fixed; bottom:20px; left:20px; width:260px; padding:15px;
                background:white; border:2px solid #333; border-radius:10px; z-index:9999; 
                font-family:Arial; font-size:12px; box-shadow:0 4px 15px rgba(0,0,0,0.3);">
        <p>Klik titik banjir untuk melihat rute</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_panel))
    
    # Add flood markers with 3 recommendations + other shelters
    flood_group = folium.FeatureGroup(name=' Titik Banjir')
    for idx, fp in flood_df.iterrows():
        flood_id = int(idx)
        flood_info = flood_point_data[flood_id]
        shelters = flood_info['shelters']
        rec = flood_info['recommendations']
        
        # Build popup with 3 recommendations + other shelters
        popup_content = f'''<div style="width:280px; max-height:380px; overflow-y:auto;">
            <h4 style="margin:0 0 8px 0; color:#FF0000;"> {flood_info['name']}</h4>
            <p style="font-size:10px; color:#666; margin:0 0 8px 0;">
                Radius exclusion: {FLOOD_RISK_RADIUS_METERS}m | Max jarak: {MAX_ROUTE_DISTANCE_KM}km
            </p>'''
        
        if not shelters:
            popup_content += '<p style="color:red;">Tidak ada shelter aman dalam radius 5km</p>'
        else:
            # 3 RECOMMENDATIONS section
            popup_content += '<div style="background:#f0f8ff; padding:8px; border-radius:6px; margin-bottom:8px;">'
            popup_content += '<p style="margin:0 0 6px 0; font-weight:bold;"> REKOMENDASI:</p>'
            
            rec_ids = [rec['tercepat'], rec['teraman'], rec['seimbang']]
            rec_labels = [' Tercepat', ' Teraman', ' Seimbang']
            rec_types = ['tercepat', 'teraman', 'seimbang']
            rec_colors = ['#e3f2fd', '#e8f5e9', '#fff8e1']
            rec_borders = ['#2196f3', '#4caf50', '#ff9800']
            
            shown_ids = set()
            for i, (rec_id, label, rtype, bg, border) in enumerate(zip(rec_ids, rec_labels, rec_types, rec_colors, rec_borders)):
                if rec_id is not None and rec_id not in shown_ids:
                    shelter = next((s for s in shelters if s['id'] == rec_id), None)
                    if shelter:
                        popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'{rtype}')" 
                            style="display:block; width:100%; margin:3px 0; padding:6px; text-align:left; 
                            background:{bg}; border:2px solid {border}; border-radius:5px; cursor:pointer; font-size:11px;">
                            <b>{label}</b>: {shelter['name'][:22]}<br>
                            <small style="color:#666;">{shelter['distance_km']}km • {shelter['travel_time_min']}mnt</small>
                        </button>'''
                        shown_ids.add(rec_id)
            
            popup_content += '</div>'
            
            # OTHER SHELTERS section
            other_shelters = [s for s in shelters if s['id'] not in shown_ids]
            if other_shelters:
                popup_content += '<p style="margin:8px 0 4px 0; font-weight:bold;"> OPSI LAINNYA:</p>'
                popup_content += '<div style="max-height:140px; overflow-y:auto;">'
                for shelter in other_shelters[:8]:
                    popup_content += f'''<button onclick="showRoute({flood_id},{shelter['id']},'')" 
                        style="display:block; width:100%; margin:2px 0; padding:5px; text-align:left; 
                        background:#f5f5f5; border:1px solid #ccc; border-radius:4px; cursor:pointer; font-size:10px;">
                         {shelter['name'][:28]}<br>
                        <small style="color:#666;">{shelter['distance_km']}km • {shelter['travel_time_min']}mnt</small>
                    </button>'''
                if len(other_shelters) > 8:
                    popup_content += f'<p style="font-size:9px; color:#666; margin:4px 0;">+{len(other_shelters)-8} shelter lainnya</p>'
                popup_content += '</div>'
        
        # Cancel button
        popup_content += '''<div style="margin-top:10px; text-align:center;">
            <button onclick="cancelSelection()" style="background:#888; color:white; padding:6px 16px; 
                    border:none; border-radius:4px; cursor:pointer; font-weight:bold;"> Cancel</button>
        </div></div>'''
        
        folium.CircleMarker(
            location=[fp['Latitude'], fp['Longitude']],
            radius=9,
            popup=folium.Popup(popup_content, max_width=320),
            color='#FF0000',
            fill=True,
            fillColor='#FF0000',
            fillOpacity=0.7,
            tooltip=f" {fp.get('Desa', 'Banjir')} - {len(shelters)} shelter"
        ).add_to(flood_group)
    
    flood_group.add_to(m)
    
    # Add shelter markers
    evac_group = folium.FeatureGroup(name=' Shelter Evakuasi')
    for idx, evac in evac_df.iterrows():
        is_safe = all(
            haversine_distance(evac['Latitude'], evac['Longitude'], fp['Latitude'], fp['Longitude']) >= FLOOD_RISK_RADIUS_METERS
            for _, fp in flood_df.iterrows()
        )
        
        name = evac.get('Nama_Tempat', evac.get('Nama', f'Shelter {idx}'))
        folium.Marker(
            location=[evac['Latitude'], evac['Longitude']],
            popup=f"<b>{name}</b><br>Status: {' Aman' if is_safe else ' Dekat banjir'}",
            icon=folium.Icon(color='green' if is_safe else 'orange', icon='home', prefix='fa'),
            tooltip=f" {name[:25]}"
        ).add_to(evac_group)
    
    evac_group.add_to(m)
    
    # Heatmap
    heat_data = [[fp['Latitude'], fp['Longitude'], 1] for _, fp in flood_df.iterrows()]
    plugins.HeatMap(heat_data, name=' Heatmap Risiko Banjir', min_opacity=0.4, radius=25, blur=20).add_to(m)
    
    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Legend
    legend = f"""
    <div style="position:fixed; bottom:20px; right:20px; width:230px; padding:12px; 
                background:white; border:2px solid #333; border-radius:10px; z-index:9999; font-size:11px;">
        <h4 style="margin:0 0 8px 0;"> Legenda</h4>
        <b>Data Satelit (Sentinel-2):</b>
        <p style="margin:2px 0;"> NDWI: Kering→Berair</p>
        <p style="margin:2px 0;"> NDMI: Kering→Lembap</p>
        <p style="margin:2px 0;"> Risiko: Rendah Sedang Tinggi</p>
        <hr style="margin:6px 0;">
        <b>Evakuasi:</b>
        <p style="margin:2px 0;"> Titik Banjir (klik untuk rute)</p>
        <p style="margin:2px 0;"> Shelter Aman</p>
        <p style="margin:2px 0;"> Shelter Dekat Banjir</p>
        <hr style="margin:6px 0;">
        <b>Jenis Rute:</b>
        <p style="margin:2px 0;"> Tercepat |  Teraman |  Seimbang</p>
        <hr style="margin:6px 0;">
        <p style="font-size:9px; color:#666;">Data: {DATE_START} - {DATE_END}</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    
    # Title
    title = """
    <h3 style="position:fixed; top:15px; left:50%; transform:translateX(-50%);
               background:white; padding:8px 18px; border-radius:8px; border:2px solid #333;
               z-index:9999; font-family:Arial; font-size:15px;">
         Peta Terintegrasi: Satelit + Evakuasi Banjir - Cilacap
    </h3>
    """
    m.get_root().html.add_child(folium.Element(title))
    
    return m


def save_integrated_map(m, filename="peta_terintegrasi_satelit.html"):
    filepath = MAPS_DIR / filename
    m.save(str(filepath))
    print(f"\n Integrated map saved: {filepath}")
    return filepath


if __name__ == "__main__":
    m = create_integrated_map()
    save_integrated_map(m)
