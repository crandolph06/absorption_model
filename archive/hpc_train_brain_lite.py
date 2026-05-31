
import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
# 1. LOAD DATA
INPUT_DIR = "outputs/single_phase/repart_parquet"
OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_lite.pkl"
SAMPLE_FRAC = 0.10 
RANDOM_SEED = 42
def train_hpc_brain():
    print(f"🚀 Starting HPC Brain Training...")
    files = glob.glob(os.path.join(INPUT_DIR, "part.*.parquet"))
    
    if not files:
        print(f"❌ No parquet files found in {INPUT_DIR}")
        return
    print(f"📂 Found {len(files)} batch files. Loading data...")
    mini_batches = []
    total_rows_seen = 0
    for i, f in enumerate(files):
        if i % 10 == 0: 
            print(f"   Processing file {i}/{len(files)}...")
        
        df_chunk = pd.read_parquet(f)
        total_rows_seen += len(df_chunk)
        if SAMPLE_FRAC < 1.0:
            df_chunk = df_chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        mini_batches.append(df_chunk)
        del df_chunk
    
    if not mini_batches:
        print("❌ No data was successfully loaded. Check your parquet files.")
        return
    # Combine small samples into one dataframe
    df = pd.concat(mini_batches, ignore_index=True)
    print(f"✅ Loaded {len(df):,} rows (sampled from {total_rows_seen:,} total rows).")
    
    del mini_batches
    # 2. PREPARE FEATURES
    print("Pre-processing features...")
    base_features = [ 
        'paa', 'ute', 'exp_ratio', 'total_pilots', 
        'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty'
    ]
    for col in base_features:
        if col not in df.columns: 
            df[col] = 0
    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)
    df = df.replace([np.inf, -np.inf], 0)
    features = base_features + ['ip_ratio', 'ip_to_stud_ratio']
    X = df[features]
    targets = [
        'wg_monthly', 'fl_monthly', 'ip_monthly', 
        'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly'
    ]
    # 3. SPLIT DATA
    X_train, X_test, df_train, df_test = train_test_split(X, df[targets], test_size=0.2, random_state=RANDOM_SEED)
    
    # 3. TRAIN MODELS 
    brain = {}
    print("\n🧠 Training Lite Models (HistGradientBoosting)...")
    for target in targets:
        if target not in df_train.columns:
            continue
            
        y_train = df_train[target]
        y_test = df_test[target]
        
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)        
        brain[target] = model
        print(f"   ✅ {target:<20} R² Score: {score:.4f}")
    # 4. SAVE MODEL
    print(f"\n💾 Saving brain to {OUTPUT_MODEL}...")
    
    # --- FIX 3: Ensure directory exists ---
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    
    joblib.dump(brain, OUTPUT_MODEL, compress=5)
    
    file_size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    print(f"🎉 Done! Final Brain Size: {file_size_mb:.2f} MB")
if __name__ == "__main__":
    train_hpc_brain()

