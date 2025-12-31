import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from pathlib import Path

from src.data_loader import load_flood_data, load_evacuation_data
from src.flood_risk import FloodRiskModel, calculate_point_flood_risk, get_humidity_weight
from src.config import FLOOD_RISK_RADIUS_METERS
import seaborn as sns
from pathlib import Path
import joblib


OUTPUT_DIR = Path("output/report")
MODELS_DIR = Path("output/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def generate_feature_importance(model, save_path):
    features = ['Jarak ke Shelter\n(meter)', 'Kelembapan\n(%)', 'Curah Hujan\n(mm)']
    importances = model.feature_importances_
    
    colors = sns.color_palette("viridis", len(features))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, importances, color=colors)
    
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontweight='bold')
    
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(importances) * 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_confusion_matrix(y_true, y_pred, save_path):
    # Map numeric to category names per PDF
    kategori_names = {1: 'Normal', 2: 'Rendah', 3: 'Sedang', 4: 'Tinggi', 5: 'Sangat Tinggi'}
    
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[kategori_names.get(l, str(l)) for l in labels],
                yticklabels=[kategori_names.get(l, str(l)) for l in labels],
                ax=ax)
    
    ax.set_xlabel('Predicted Risk Level', fontsize=12)
    ax.set_ylabel('Actual Risk Level', fontsize=12)
    ax.set_title('Confusion Matrix - Flood Risk Classification', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_accuracy_chart(train_acc, test_acc, save_path):
    categories = ['Training\nAccuracy', 'Testing\nAccuracy']
    accuracies = [train_acc * 100, test_acc * 100]
    
    colors = ['#2ecc71', '#3498db']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Threshold (80%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_risk_distribution(predictions, save_path):
    # Category names per PDF BMKG rules
    kategori_names = {1: 'Normal', 2: 'Rendah', 3: 'Sedang', 4: 'Tinggi', 5: 'Sangat Tinggi'}
    
    unique, counts = np.unique(predictions, return_counts=True)
    labels = [kategori_names.get(u, str(u)) for u in unique]
    
    colors = sns.color_palette("RdYlGn_r", len(unique))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%',
                                       colors=colors, explode=[0.02]*len(unique),
                                       shadow=True, startangle=90)
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribusi Kategori Risiko Banjir (BMKG)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 50)
    print("GENERATING REPORT VISUALIZATIONS")
    print("=" * 50)
    
    # Load complete dataset with all 5 risk levels
    print("\nLoading complete training dataset...")
    data_path = "Banjir Cilacap/Dataset_Latih_Final_Lengkap.xlsx"
    df = pd.read_excel(data_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean data - handle missing values
    df = df.dropna(subset=['Jarak_KM', 'Kelembapan', 'Curah_Hujan', 'Bobot_Risiko'])
    
    # Create features: [Jarak_KM (meters), Kelembapan, Curah_Hujan]
    X = np.column_stack([
        df['Jarak_KM'].values * 1000,  # Convert to meters
        df['Kelembapan'].values,
        df['Curah_Hujan'].values       # New feature!
    ])
    
    # Labels are already 1-5 in Bobot_Risiko
    y = df['Bobot_Risiko'].astype(int).values
    
    print(f"\nFeatures: [Jarak (m), Kelembapan (%), Curah Hujan (mm)]")
    print(f"Features shape: {X.shape}")
    print(f"Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Humidity range: {X[:,1].min():.1f}% - {X[:,1].max():.1f}%")
    print(f"Rainfall range: {X[:,2].min():.1f} - {X[:,2].max():.1f} mm")
    
    # Train model
    print("\nTraining RandomForest model...")
    model = FloodRiskModel()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.model.fit(X_train, y_train)
    model.is_trained = True
    
    train_acc = model.model.score(X_train, y_train)
    test_acc = model.model.score(X_test, y_test)
    y_pred = model.model.predict(X_test)
    
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Kappa Score:    {kappa:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    generate_feature_importance(
        model.model, 
        OUTPUT_DIR / "feature_importance.png"
    )
    
    generate_confusion_matrix(
        y_test, y_pred,
        OUTPUT_DIR / "confusion_matrix.png"
    )
    
    generate_accuracy_chart(
        train_acc, test_acc,
        OUTPUT_DIR / "accuracy_comparison.png"
    )
    
    # Predict risk for all points
    all_predictions = model.model.predict(X)
    generate_risk_distribution(
        all_predictions,
        OUTPUT_DIR / "risk_distribution.png"
    )
    
    # Save metrics to text file
    metrics_path = OUTPUT_DIR / "model_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("=" * 40 + "\n")
        f.write("FLOOD RISK MODEL EVALUATION METRICS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Testing Samples:  {len(X_test)}\n\n")
        f.write(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)\n")
        f.write(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)\n")
        f.write(f"Cohen's Kappa:     {kappa:.4f}\n\n")
        f.write("Feature Importance:\n")
        features = ['jarak_km', 'kelembapan', 'curah_hujan']
        for feat, imp in zip(features, model.model.feature_importances_):
            f.write(f"  {feat}: {imp:.4f}\n")
        f.write("\n" + "=" * 40 + "\n")
        f.write("Classification Report:\n")
        f.write("=" * 40 + "\n")
        kategori_names = {1: 'Normal', 2: 'Rendah', 3: 'Sedang', 4: 'Tinggi', 5: 'Sangat Tinggi'}
        unique_labels = sorted(list(set(y_test) | set(y_pred)))
        target_names = [kategori_names.get(l, str(l)) for l in unique_labels]
        f.write(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"Saved: {metrics_path}")
    
    # ============================================
    # SAVE MODEL
    # ============================================
    model_path = MODELS_DIR / "rf_risk_model.pkl"
    # Access the internal sklearn model from FloodRiskModel wrapper
    joblib.dump(model.model, model_path)
    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 50)
    print("ALL VISUALIZATIONS GENERATED!")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
