# Sistem Rekomendasi Rute Evakuasi Banjir - Cilacap

Sistem rekomendasi rute evakuasi berbasis peta digital untuk BPBD Kabupaten Cilacap.

## Fitur Utama

- **Routing Cerdas**: Rute optimal dengan mempertimbangkan risiko banjir
- **Model Random Forest**: Prediksi tingkat risiko berdasarkan jarak dan kelembapan
- **Peta Interaktif**: Visualisasi dengan Folium (heatmap, marker, rute)
- **Simulasi Cuaca**: Kondisi normal hingga risiko sangat tinggi

## Instalasi

```bash
pip install -r requirements.txt
```

## Penggunaan

```bash
# Semua simulasi
python main.py --mode all

# Kondisi normal saja
python main.py --mode normal

# Kondisi banjir dengan 5 rute acak
python main.py --mode flood --simulations 5
```

## Data yang Digunakan

| File | Deskripsi |
|------|-----------|
| Data_Banjir_FIX_120.xlsx | 120 titik potensi banjir |
| Data_Evakuasi_FIX_314.xlsx | 314 lokasi evakuasi |
| Data_Waktu_Tempuh_Final_Fixed.xlsx | Waktu tempuh antar lokasi |
| Stasiun Metalurgi Tunggul Wulung.xlsx | Data kelembapan & curah hujan |

## Pembobotan Kelembapan

| Kelembapan (%) | Kategori | Bobot |
|----------------|----------|-------|
| ≤ 70 | Normal | 1 |
| 71–80 | Rendah | 2 |
| 81–90 | Sedang | 3 |
| 91–95 | Tinggi | 4 |
| > 95 | Sangat Tinggi | 5 |

## Output

Peta HTML interaktif tersimpan di folder `output/maps/`:
- `simulasi_normal.html` - Rute kondisi normal
- `simulasi_banjir.html` - Rute kondisi banjir
- `peta_lengkap.html` - Semua layer data

## Teknologi

- OSMnx - Jaringan jalan OpenStreetMap
- Folium - Peta interaktif
- Scikit-learn - Model Random Forest
- GeoPandas - Data geospasial

## Lisensi

BPBD Kabupaten Cilacap - 2024
