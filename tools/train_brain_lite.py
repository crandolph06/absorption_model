import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib

# 1. LOAD DATA
path = "outputs/simulation_results.parquet" # Ensure this path is correct
print(f"📂 Loading {path}...")
df = pd.read_parquet(path)

# Fill missing columns
features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
for col in features:
    if col not in df.columns: df[col] = 0

# 2. TRAIN LIGHTWEIGHT MODELS
targets = [
    'wg_monthly', 'fl_monthly', 'ip_monthly', 
    'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly'
]
models = {}

print("🧠 Training Lite Models...")
for target in targets:
    # Filter clean data
    clean_df = df[df[target] >= 0.0] 
    X = clean_df[features]
    y = clean_df[target]
    
    # ⚡ OPTIMIZATION SETTINGS ⚡
    model = HistGradientBoostingRegressor(
        max_iter=300,           # High iterations to refine the "cliffs"
        max_depth=10,           # 10 is the "Sweet Spot." Deep enough for physics, shallow enough for RAM.
        min_samples_leaf=10,    # Small enough to catch the "Surge" dots you saw
        l2_regularization=0.1,  # Keeps the curves smooth (prevents jagged zig-zags)
        learning_rate=0.1,      # Standard learning rate for stability
        random_state=42
    )
    model.fit(X, y)
    
    models[target] = model
    print(f"   - {target} Score: {model.score(X, y):.4f}")

# 3. SAVE WITH COMPRESSION
filename = "sortie_brain.pkl"
# compress=3 drastically reduces file size
joblib.dump(models, filename, compress=3) 
print(f"\n✅ Lite Brain saved to {filename}")