
import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

INPUT_DIR = "outputs/single_phase/repart_parquet"
OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_hybrid.pkl" 
SAMPLE_FRAC = 0.10 
RANDOM_SEED = 42

def train_hpc_multi_brain():
    print(f"🚀 Starting Multi-Output HPC Brain Training...")
    files = glob.glob(os.path.join(INPUT_DIR, "part.*.parquet"))
    
    if not files:
        print(f"❌ No parquet files found in {INPUT_DIR}")
        return

    # 1. LOAD & SAMPLE DATA
    mini_batches = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        if SAMPLE_FRAC < 1.0:
            df_chunk = df_chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        mini_batches.append(df_chunk)
    
    df = pd.concat(mini_batches, ignore_index=True)
    print(f"📊 Dataset loaded: {len(df):,} rows")

    # 2. FEATURE & TARGET SELECTION
    base_features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
    for col in base_features:
        if col not in df.columns: 
            df[col] = 0
    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)
    
    df = df.replace([np.inf, -np.inf], 0)
    features = base_features + ['ip_ratio', 'ip_to_stud_ratio']
    
    targets = [
        'wg_monthly', 'fl_monthly', 'ip_monthly', 
        'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly'
    ]

    X = df[features].fillna(0)
    Y = df[targets].fillna(0)

    # 3. SPLIT
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

    # # 4. DEFINE HYBRID MODEL
    base_model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    linear_model = MultiOutputRegressor(base_model)
    print("🧠 Training Linear model...")
    linear_model.fit(X_train, Y_train)

    Y_train_pred_lin = linear_model.predict(X_train)
    residuals = Y_train - Y_train_pred_lin

    booster_model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.03,     
        max_leaf_nodes=15,      
        min_samples_leaf=150,   
        l2_regularization=10.0,  
        random_state=RANDOM_SEED
        )
    )
    print("🧠 Training Booster model...")
    booster_model.fit(X_test, residuals)

    # 5. EVALUATE
    y_pred_lin_test = linear_model.predict(X_test)
    y_pred_res_test = booster_model.predict(X_test)
    y_pred_total = y_pred_lin_test + y_pred_res_test

    score = r2_score(Y_test, y_pred_total)
    print(f"✅ Training Complete. Overall R² Score: {score:.4f}")

    # 6. SAVE
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    hybrid_brain = {
        'linear': linear_model,
        'booster': booster_model
    }
    joblib.dump(hybrid_brain, OUTPUT_MODEL)
    print(f"💾 Hybrid brain saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    train_hpc_multi_brain()