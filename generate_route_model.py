import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.tree import plot_tree
import seaborn as sns
import joblib
from pathlib import Path

OUTPUT_DIR = Path("output/report")
MODELS_DIR = Path("output/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.size'] = 10


def create_route_labels(df):
    """
    Untuk setiap Desa_Asal, tentukan shelter mana yang:
    - Tercepat: Waktu_Tempuh_Menit minimum
    - Teraman: Bobot_Risiko minimum
    - Seimbang: kombinasi keduanya
    """
    labels = []
    
    # Group by Desa_Asal untuk menentukan label relatif
    for desa in df['Desa_Asal'].unique():
        desa_data = df[df['Desa_Asal'] == desa].copy()
        
        if len(desa_data) == 0:
            continue
            
        # Normalize scores (0-1)
        waktu_min = desa_data['Waktu_Tempuh_Menit'].min()
        waktu_max = desa_data['Waktu_Tempuh_Menit'].max()
        risiko_min = desa_data['Bobot_Risiko'].min()
        risiko_max = desa_data['Bobot_Risiko'].max()
        
        for idx, row in desa_data.iterrows():
            # Calculate normalized scores (lower is better)
            if waktu_max > waktu_min:
                speed_score = (row['Waktu_Tempuh_Menit'] - waktu_min) / (waktu_max - waktu_min)
            else:
                speed_score = 0
                
            if risiko_max > risiko_min:
                safety_score = (row['Bobot_Risiko'] - risiko_min) / (risiko_max - risiko_min)
            else:
                safety_score = 0
            
            # Combined score for balance
            balance_score = 0.5 * speed_score + 0.5 * safety_score
            
            # Determine label based on which aspect this route excels at
            if speed_score <= 0.33 and safety_score > 0.5:
                label = 0  # Tercepat
            elif safety_score <= 0.33 and speed_score > 0.5:
                label = 1  # Teraman
            elif balance_score <= 0.4:
                label = 2  # Seimbang
            elif speed_score < safety_score:
                label = 0  # Tercepat
            elif safety_score < speed_score:
                label = 1  # Teraman
            else:
                label = 2  # Seimbang
                
            labels.append((idx, label))
    
    return dict(labels)


def main():
    print("=" * 60)
    print("ROUTE SELECTION MODEL - Random Forest")
    print("=" * 60)
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_excel("Banjir Cilacap/Dataset_Latih_Final_Lengkap.xlsx")
    print(f"Loaded {len(df)} samples")
    
    # Clean data
    df = df.dropna(subset=['Jarak_KM', 'Waktu_Tempuh_Menit', 'Bobot_Risiko', 'Kelembapan', 'Curah_Hujan'])
    print(f"After cleaning: {len(df)} samples")
    
    # Create labels
    print("\nCreating route type labels...")
    label_dict = create_route_labels(df)
    df['Route_Type'] = df.index.map(label_dict)
    df = df.dropna(subset=['Route_Type'])
    df['Route_Type'] = df['Route_Type'].astype(int)
    
    print(f"Label distribution:")
    label_names = {0: 'Tercepat', 1: 'Teraman', 2: 'Seimbang'}
    for label, name in label_names.items():
        count = len(df[df['Route_Type'] == label])
        print(f"  {name}: {count}")
    
    # Features
    feature_cols = ['Jarak_KM', 'Waktu_Tempuh_Menit', 'Bobot_Risiko', 'Kelembapan', 'Curah_Hujan']
    X = df[feature_cols].values
    y = df['Route_Type'].values
    
    print(f"\nFeatures: {feature_cols}")
    print(f"X shape: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Increased from 5 to 10 for better accuracy
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    y_pred = rf_model.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Testing Accuracy:  {test_acc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    
    # Feature Importance
    print("\nFeature Importance:")
    for feat, imp in zip(feature_cols, rf_model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    # ============================================
    # VISUALIZATION 1: Decision Tree
    # ============================================
    print("\nGenerating Decision Tree visualization...")
    # Larger figure size with better proportions to prevent overlapping
    fig, ax = plt.subplots(figsize=(35, 18))
    plot_tree(
        rf_model.estimators_[0],  # First tree
        feature_names=feature_cols,
        class_names=['Tercepat', 'Teraman', 'Seimbang'],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
        max_depth=3,  # Reduced to 3 for cleaner visualization
        proportion=True,  # Show proportions instead of raw counts
        impurity=False   # Hide impurity for cleaner look
    )
    plt.title('Decision Tree - Route Selection (Tree #1 from Random Forest)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout(pad=3.0)  # Add padding
    plt.savefig(OUTPUT_DIR / 'decision_tree_route.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'decision_tree_route.png'}")
    
    # ============================================
    # VISUALIZATION 2: Feature Importance
    # ============================================
    print("Generating Feature Importance chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    features_display = ['Jarak\n(KM)', 'Waktu Tempuh\n(Menit)', 'Bobot\nRisiko', 'Kelembapan\n(%)', 'Curah Hujan\n(mm)']
    colors = sns.color_palette("viridis", len(features_display))
    bars = ax.barh(features_display, rf_model.feature_importances_, color=colors)
    for bar, imp in zip(bars, rf_model.feature_importances_):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance - Route Selection Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_route.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'feature_importance_route.png'}")
    
    # ============================================
    # VISUALIZATION 3: Confusion Matrix
    # ============================================
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tercepat', 'Teraman', 'Seimbang'],
                yticklabels=['Tercepat', 'Teraman', 'Seimbang'],
                ax=ax)
    ax.set_xlabel('Predicted Route Type')
    ax.set_ylabel('Actual Route Type')
    ax.set_title('Confusion Matrix - Route Selection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_route.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix_route.png'}")
    
    # ============================================
    # VISUALIZATION 4: Learning Curve
    # ============================================
    print("Generating Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        rf_model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training score
    ax.plot(train_sizes, train_mean, 'o-', color='#2ecc71', linewidth=2, 
            label='Training Accuracy', markersize=8)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#2ecc71')
    
    # Plot validation score
    ax.plot(train_sizes, test_mean, 'o-', color='#3498db', linewidth=2,
            label='Validation Accuracy (CV=5)', markersize=8)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color='#3498db')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curve - Route Selection Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'learning_curve_route.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'learning_curve_route.png'}")
    
    # ============================================
    # Save metrics
    # ============================================
    metrics_path = OUTPUT_DIR / 'route_model_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("ROUTE SELECTION MODEL - EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Testing Samples:  {len(X_test)}\n\n")
        f.write(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)\n")
        f.write(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)\n")
        f.write(f"Cohen's Kappa:     {kappa:.4f}\n\n")
        f.write("Feature Importance:\n")
        for feat, imp in zip(feature_cols, rf_model.feature_importances_):
            f.write(f"  {feat}: {imp:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Classification Report:\n")
        f.write("=" * 50 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=['Tercepat', 'Teraman', 'Seimbang']))
    print(f"Saved: {metrics_path}")
    
    # ============================================
    # Save Model
    # ============================================
    model_path = MODELS_DIR / "rf_route_model.pkl"
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("ALL ROUTE SELECTION VISUALIZATIONS GENERATED!")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
