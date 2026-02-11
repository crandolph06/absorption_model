import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
INPUT_DIR = "outputs/single_phase/parquet" # For HPC
# INPUT_DIR = "outputs/hpc" # For Debugging 
OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_lite.pkl"
SAMPLE_FRAC = 0.10 # Consider more than 10%... we'll see if this works...
RANDOM_SEED = 42

def train_hpc_brain():
    print(f"🚀 Starting HPC Brain Training...")

    # 1. LOAD DATA
    files = glob.glob(os.path.join(INPUT_DIR, "batch_*.parquet"))
    
    if not files:
        print(f"❌ No parquet files found in {INPUT_DIR}")
        return

    print(f"📂 Found {len(files)} batch files. Loading data...")

    mini_batches = []
    total_rows_seen = 0

    for i, f in enumerate(files):
        if i % 50 == 0: 
            print(f"   Processing file {i}/{len(files)}...", end='\r')
        
        df_chunk = pd.read_parquet(f)
        total_rows_seen += len(df_chunk)

        if SAMPLE_FRAC < 1.0:
            df_chunk = df_chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)

        mini_batches.append(df_chunk)

        del df_chunk
    
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
        if col not in df.columns: df[col] = 0

    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']

    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)

    df.replace([np.inf, -np.inf], 0, inplace=True)

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
        if target not in df.columns:
            continue
            
        y_train = df_train[target]
        y_test = df_test[target]
        
        model = HistGradientBoostingRegressor(
            max_iter=200,          # Sufficient for convergence on smooth data
            max_depth=10,          # Limits complexity/filesize
            learning_rate=0.1,
            l2_regularization=0.1, # Prevents overfitting on noise
            random_state=RANDOM_SEED,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)        
        brain[target] = model
        print(f"   ✅ {target:<20} R² Score: {score:.4f}")

    # 4. SAVE MODEL
    print(f"\n💾 Saving brain to {OUTPUT_MODEL}...")
    
    # compress=5 is a good balance of size vs load speed
    joblib.dump(brain, OUTPUT_MODEL, compress=5)
    
    file_size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    print(f"🎉 Done! Final Brain Size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 100:
        print("⚠️ Warning: File is larger than 100MB. Consider reducing max_iter or max_depth.")

if __name__ == "__main__":
    train_hpc_brain()
