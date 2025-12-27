"""
Flood Evacuation Route Recommendation System - Cilacap
=======================================================

Main entry point for the flood evacuation routing system.
Supports multiple simulation modes for BPBD Cilacap.

Usage:
    python main.py --mode normal    # Normal condition simulation
    python main.py --mode flood     # Flood condition simulation
    python main.py --mode all       # Generate all maps and simulations
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CILACAP_CENTER, MAPS_DIR, get_humidity_weight, FLOOD_RISK_RADIUS_METERS, MAX_ROUTE_DISTANCE_KM
from src.data_loader import load_flood_data, load_evacuation_data, load_weather_data, load_travel_time_data
from src.flood_risk import FloodRiskModel, calculate_point_flood_risk
from src.routing import load_road_network, find_flood_aware_route, find_nearest_safe_evacuation
from src.visualization import (
    create_complete_map, save_map, create_base_map,
    add_flood_markers, add_evacuation_markers, add_flood_heatmap,
    add_route, add_route_info_box, add_legend
)
from src.weather import (
    get_current_weather_simulation, simulate_normal_condition,
    simulate_high_risk_condition, generate_random_origin,
    get_simulation_scenario
)
from src.interactive_map import create_interactive_flood_map, save_interactive_map

import folium


def print_header():
    """Print application header."""
    print("=" * 60)
    print("  SISTEM REKOMENDASI RUTE EVAKUASI BANJIR - CILACAP")
    print("  BPBD Kabupaten Cilacap")
    print("=" * 60)
    print()


def run_normal_simulation():
    """
    Run normal condition simulation.
    Route from Alun-alun Cilacap to nearest safe evacuation point.
    """
    print("\n" + "=" * 50)
    print("SIMULASI 1: KONDISI NORMAL")
    print("Rute dari Alun-alun Cilacap ke Titik Evakuasi Terdekat")
    print("=" * 50)
    
    # Load data
    print("\n[1/5] Memuat data...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    
    # Get normal humidity
    humidity, desc = simulate_normal_condition()
    print(f"\n[2/5] Kondisi Cuaca: {desc}")
    print(f"      Kelembapan: {humidity:.1f}%")
    
    humidity_weight, humidity_label = get_humidity_weight(humidity)
    print(f"      Tingkat Risiko: {humidity_label} ({humidity_weight})")
    
    # Load road network
    print("\n[3/5] Memuat jaringan jalan...")
    G = load_road_network()
    
    # Find route to nearest safe evacuation
    print("\n[4/5] Menghitung rute optimal...")
    origin = CILACAP_CENTER  # Alun-alun Cilacap
    
    route_info = find_nearest_safe_evacuation(
        G, origin, evac_df, flood_df,
        humidity=humidity, max_risk_level=3
    )
    
    if route_info['success']:
        print(f"\n      ✓ Rute ditemukan!")
        print(f"      Tujuan: {route_info.get('evacuation_name', 'Evakuasi')}")
        print(f"      Jarak: {route_info['distance_km']:.2f} km")
        print(f"      Waktu Tempuh: {route_info['travel_time_min']:.1f} menit")
    else:
        print(f"\n      ✗ Rute tidak ditemukan: {route_info.get('error', 'Unknown error')}")
    
    # Create map
    print("\n[5/5] Membuat peta interaktif...")
    m = create_complete_map(
        flood_df, evac_df, route_info,
        humidity=humidity,
        title="Simulasi Normal - Alun-alun Cilacap ke Evakuasi"
    )
    
    map_path = save_map(m, "simulasi_normal.html")
    print(f"\n✓ Peta disimpan: {map_path}")
    
    return route_info


def run_flood_simulation(num_simulations: int = 3):
    """
    Run flood condition simulation.
    Random points to safe evacuation avoiding high-risk areas.
    """
    print("\n" + "=" * 50)
    print("SIMULASI 2: KONDISI BANJIR")
    print("Rute Acak ke Titik Evakuasi (Menghindari Area Rawan)")
    print("=" * 50)
    
    # Load data
    print("\n[1/5] Memuat data...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    
    # Get high flood risk humidity
    humidity, desc = simulate_high_risk_condition()
    print(f"\n[2/5] Kondisi Cuaca: {desc}")
    print(f"      Kelembapan: {humidity:.1f}%")
    
    humidity_weight, humidity_label = get_humidity_weight(humidity)
    print(f"      Tingkat Risiko: {humidity_label} ({humidity_weight})")
    
    # Load road network
    print("\n[3/5] Memuat jaringan jalan...")
    G = load_road_network()
    
    # Create base map
    m = create_base_map()
    
    # Add heatmap and markers
    m = add_flood_heatmap(m, flood_df, humidity)
    m = add_flood_markers(m, flood_df, humidity)
    m = add_evacuation_markers(m, evac_df, flood_df, humidity)
    
    # Run multiple random simulations
    print(f"\n[4/5] Menjalankan {num_simulations} simulasi acak...")
    
    route_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF']
    all_routes = []
    
    for i in range(num_simulations):
        print(f"\n      Simulasi {i+1}/{num_simulations}:")
        
        # Generate random origin
        origin = generate_random_origin(flood_df, evac_df)
        print(f"      Titik Awal: {origin[0]:.6f}, {origin[1]:.6f}")
        
        # Find route
        route_info = find_nearest_safe_evacuation(
            G, origin, evac_df, flood_df,
            humidity=humidity, max_risk_level=2  # Only very safe evacuations
        )
        
        if route_info['success']:
            print(f"      ✓ Rute ke: {route_info.get('evacuation_name', 'Evakuasi')[:30]}...")
            print(f"      Jarak: {route_info['distance_km']:.2f} km, Waktu: {route_info['travel_time_min']:.1f} menit")
            
            # Add route to map with different color
            route_color = route_colors[i % len(route_colors)]
            m = add_route(m, route_info, color=route_color)
            all_routes.append(route_info)
        else:
            print(f"      ✗ Tidak ada rute aman dari titik ini")
    
    # Add legend
    m = add_legend(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px; font-family: Arial;">
        <b>Simulasi Banjir - {num_simulations} Rute Acak</b><br>
        <small>Kelembapan: {humidity:.1f}% ({humidity_label})</small>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    print("\n[5/5] Membuat peta interaktif...")
    map_path = save_map(m, "simulasi_banjir.html")
    print(f"\n✓ Peta disimpan: {map_path}")
    
    return all_routes


def run_all_layers_map():
    """
    Generate map with all data layers.
    """
    print("\n" + "=" * 50)
    print("PETA LENGKAP: SEMUA LAYER DATA")
    print("=" * 50)
    
    # Load all data
    print("\n[1/3] Memuat semua data...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    weather_data = get_current_weather_simulation()
    
    avg_humidity = weather_data['humidity']['avg']
    humidity_weight = weather_data['humidity']['weight']
    humidity_label = weather_data['humidity']['label']
    
    print(f"      Rata-rata Kelembapan: {avg_humidity:.1f}%")
    print(f"      Tingkat Risiko: {humidity_label}")
    
    # Create complete map
    print("\n[2/3] Membuat peta dengan semua layer...")
    m = create_complete_map(
        flood_df, evac_df,
        humidity=avg_humidity,
        title=f"Peta Evakuasi Banjir Cilacap - Semua Layer"
    )
    
    # Save map
    print("\n[3/3] Menyimpan peta...")
    map_path = save_map(m, "peta_lengkap.html")
    print(f"\n✓ Peta disimpan: {map_path}")
    
    return map_path


def run_model_training():
    """
    Train and evaluate the Random Forest flood risk model.
    """
    print("\n" + "=" * 50)
    print("TRAINING MODEL RANDOM FOREST")
    print("=" * 50)
    
    # Load data
    print("\n[1/3] Memuat data training...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    weather_df = load_weather_data()
    
    # Train model
    print("\n[2/3] Melatih model...")
    model = FloodRiskModel()
    metrics = model.train(flood_df, evac_df, weather_df)
    
    if metrics.get('status') == 'success':
        print("\n[3/3] Hasil Training:")
        print(f"      Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"      Test Accuracy: {metrics['test_accuracy']:.2%}")
        print("\n      Feature Importance:")
        for feature, importance in metrics['feature_importance'].items():
            print(f"        - {feature}: {importance:.3f}")
    else:
        print(f"\n      Status: {metrics.get('status', 'unknown')}")
    
    return metrics


def run_interactive_map():
    """
    Generate interactive map where clicking flood points shows available routes.
    """
    print("\n" + "=" * 50)
    print("PETA INTERAKTIF: KLIK TITIK BANJIR UNTUK RUTE")
    print("=" * 50)
    
    # Load data
    print("\n[1/4] Memuat data...")
    flood_df = load_flood_data()
    evac_df = load_evacuation_data()
    weather_data = get_current_weather_simulation()
    
    avg_humidity = weather_data['humidity']['avg']
    humidity_label = weather_data['humidity']['label']
    
    print(f"\n[2/4] Konfigurasi:")
    print(f"      Radius exclusion shelter: {FLOOD_RISK_RADIUS_METERS}m")
    print(f"      Max jarak rute: {MAX_ROUTE_DISTANCE_KM}km")
    print(f"      Kelembapan: {avg_humidity:.1f}% ({humidity_label})")
    
    # Count safe shelters
    print("\n[3/4] Menghitung shelter aman per titik banjir...")
    total_flood = len(flood_df)
    total_evac = len(evac_df)
    
    print(f"      Total titik banjir: {total_flood}")
    print(f"      Total shelter: {total_evac}")
    
    # Create interactive map
    print("\n[4/4] Membuat peta interaktif...")
    m = create_interactive_flood_map(flood_df, evac_df, humidity=avg_humidity)
    
    # Save map
    map_path = save_interactive_map(m, "peta_interaktif.html")
    
    print(f"\n✓ Peta interaktif disimpan: {map_path}")
    print("\n  CARA PENGGUNAAN:")
    print("  1. Buka file HTML di browser")
    print("  2. Klik titik banjir (merah) untuk melihat shelter tersedia")
    print("  3. Pilih shelter tujuan dari daftar")
    print("  4. Rute akan ditampilkan otomatis")
    
    return map_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sistem Rekomendasi Rute Evakuasi Banjir - Cilacap'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['normal', 'flood', 'all', 'train', 'layers', 'interactive'],
        default='interactive',
        help='Simulation mode (default: interactive)'
    )
    parser.add_argument(
        '--simulations', '-s',
        type=int,
        default=3,
        help='Number of random simulations for flood mode (default: 3)'
    )
    parser.add_argument(
        '--humidity', '-rh',
        type=float,
        default=None,
        help='Set humidity percentage for simulation (default: from weather data)'
    )
    
    args = parser.parse_args()
    
    print_header()
    
    start_time = time.time()
    
    if args.mode == 'normal':
        run_normal_simulation()
    
    elif args.mode == 'flood':
        run_flood_simulation(args.simulations)
    
    elif args.mode == 'train':
        metrics = run_model_training()
        # Display accuracy prominently
        if metrics.get('status') == 'success':
            print("\n" + "=" * 50)
            print("📊 AKURASI MODEL RANDOM FOREST")
            print("=" * 50)
            print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
            print(f"   Test Accuracy:  {metrics['test_accuracy']:.2%}")
            print("=" * 50)
    
    elif args.mode == 'layers':
        run_all_layers_map()
    
    elif args.mode == 'interactive':
        # Default mode - interactive map
        run_interactive_map()
    
    elif args.mode == 'all':
        # Run all simulations
        print("\nMenjalankan semua simulasi...\n")
        
        # 1. Train model
        metrics = run_model_training()
        
        # Display accuracy prominently
        if metrics.get('status') == 'success':
            print("\n" + "=" * 50)
            print("📊 AKURASI MODEL RANDOM FOREST")
            print("=" * 50)
            print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
            print(f"   Test Accuracy:  {metrics['test_accuracy']:.2%}")
            print("=" * 50)
        
        # 2. Interactive map (main feature)
        run_interactive_map()
        
        # 3. Normal simulation
        run_normal_simulation()
        
        # 4. Flood simulation
        run_flood_simulation(args.simulations)
        
        # 5. All layers map
        run_all_layers_map()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Selesai! Total waktu: {elapsed:.1f} detik")
    print(f"Output maps tersimpan di: {MAPS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

