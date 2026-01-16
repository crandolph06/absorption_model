import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib  # Used to save the "Brain" to a file

# 1. LOAD DATA
path = "outputs/simulation_results.parquet"
print(f"📂 Loading {path}...")
df = pd.read_parquet(path)

# Fill missing columns with 0
features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
for col in features:
    if col not in df.columns: df[col] = 0

# 2. TRAIN 3 SEPARATE MODELS
targets = ['wg_monthly', 'fl_monthly', 'ip_monthly', 'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly']
models = {}

print("🧠 Training Neural Models (Random Forest)...")
for target in targets:
    print(f"   - Learning physics for {target}...")
    
    # Filter out garbage (Optional: keeps the training clean)
    clean_df = df[df[target] > 0.1]
    
    X = clean_df[features]
    y = clean_df[target]
    
    # Higher estimators = smoother curves in your app
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y)
    
    models[target] = model
    print(f"     Score: {model.score(X, y):.4f}")

# 3. SAVE THE BRAIN
filename = "sortie_brain.pkl"
joblib.dump(models, filename)
print(f"\n✅ Brain saved to {filename}")
print("   Move this file to your Streamlit app folder.")