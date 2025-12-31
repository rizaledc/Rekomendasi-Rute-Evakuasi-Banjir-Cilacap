import ee
import folium
from folium import plugins
import json
from typing import Dict, Tuple, Optional
from pathlib import Path

from .config import CILACAP_BOUNDS, CILACAP_CENTER, MAPS_DIR


SENTINEL_BANDS = {
    'blue': 'B2',
    'green': 'B3',
    'red': 'B4',
    'nir': 'B8',
    'swir1': 'B11',
    'swir2': 'B12',
}

DATE_START = '2024-10-01'
DATE_END = '2024-12-16'

GEE_PROJECT_ID = 'tugasbesarcilacap'


def initialize_ee(project: str = GEE_PROJECT_ID):
    try:
        ee.Initialize(project=project)
        print(f"Google Earth Engine initialized with project: {project}")
        return True
    except Exception as e:
        print("GEE initialization failed.")
        print("Please run: earthengine authenticate")
        print(f"Error: {e}")
        return False


def get_cilacap_geometry() -> ee.Geometry:
    return ee.Geometry.Rectangle([
        CILACAP_BOUNDS['west'],
        CILACAP_BOUNDS['south'],
        CILACAP_BOUNDS['east'],
        CILACAP_BOUNDS['north']
    ])


def get_sentinel2_composite(
    start_date: str = DATE_START,
    end_date: str = DATE_END,
    cloud_threshold: int = 20
) -> ee.Image:
    geometry = get_cilacap_geometry()
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    )
    
    count = collection.size().getInfo()
    print(f"Found {count} Sentinel-2 images for the date range")
    
    if count == 0:
        raise ValueError(f"No images found between {start_date} and {end_date}")
    
    composite = collection.median().clip(geometry)
    return composite


def calculate_ndwi(image: ee.Image) -> ee.Image:
    green = image.select(SENTINEL_BANDS['green'])
    nir = image.select(SENTINEL_BANDS['nir'])
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    return ndwi


def calculate_ndmi(image: ee.Image) -> ee.Image:
    nir = image.select(SENTINEL_BANDS['nir'])
    swir = image.select(SENTINEL_BANDS['swir1'])
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename('NDMI')
    return ndmi


def classify_water(ndwi: ee.Image, threshold: float = 0.3) -> ee.Image:
    water_mask = ndwi.gt(threshold).selfMask().rename('water_mask')
    return water_mask


def classify_moisture(ndmi: ee.Image) -> ee.Image:
    dry = ndmi.lt(0).multiply(1)
    moderate = ndmi.gte(0).And(ndmi.lt(0.2)).multiply(2)
    wet = ndmi.gte(0.2).And(ndmi.lt(0.4)).multiply(3)
    very_wet = ndmi.gte(0.4).multiply(4)
    moisture_class = dry.add(moderate).add(wet).add(very_wet).rename('moisture_class')
    return moisture_class


def get_tile_url(image: ee.Image, vis_params: Dict) -> str:
    map_id = image.getMapId(vis_params)
    return map_id['tile_fetcher'].url_format


def create_satellite_map(
    start_date: str = DATE_START,
    end_date: str = DATE_END
) -> folium.Map:
    print(f"Creating satellite analysis map ({start_date} to {end_date})")
    
    if not initialize_ee():
        raise RuntimeError("Failed to initialize Google Earth Engine")
    
    print("Fetching Sentinel-2 composite")
    composite = get_sentinel2_composite(start_date, end_date)
    
    print("Calculating NDWI")
    ndwi = calculate_ndwi(composite)
    
    print("Calculating NDMI")
    ndmi = calculate_ndmi(composite)
    
    water = classify_water(ndwi)
    moisture = classify_moisture(ndmi)
    
    m = folium.Map(location=CILACAP_CENTER, zoom_start=10, tiles='OpenStreetMap')
    
    ndwi_url = get_tile_url(ndwi, {
        'min': -1, 'max': 1,
        'palette': ['red', 'yellow', 'green', 'cyan', 'blue']
    })
    
    ndmi_url = get_tile_url(ndmi, {
        'min': -0.5, 'max': 0.5,
        'palette': ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
    })
    
    water_url = get_tile_url(water, {'palette': ['0000FF']})
    
    folium.TileLayer(tiles=ndwi_url, attr='NDWI', name='NDWI (Water)', overlay=True, show=False).add_to(m)
    folium.TileLayer(tiles=ndmi_url, attr='NDMI', name='NDMI (Moisture)', overlay=True, show=False).add_to(m)
    folium.TileLayer(tiles=water_url, attr='Water', name='Water Mask', overlay=True, show=True).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    legend_html = '''
    <div style="position:fixed; top:10px; right:10px; width:180px; padding:10px; 
    background:white; border:2px solid #333; border-radius:10px; z-index:9999;">
        <h4>Legenda</h4>
        <p>NDWI: Kering - Berair</p>
        <p>NDMI: Kering - Lembap</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def run_satellite_analysis(output_filename: str = "peta_satelit_analisis.html"):
    print("Running satellite-based flood risk analysis")
    
    try:
        m = create_satellite_map()
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MAPS_DIR / output_filename
        m.save(str(output_path))
        print(f"Satellite analysis map saved: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Satellite analysis failed: {e}")
        return None


if __name__ == "__main__":
    run_satellite_analysis()
