
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "outputs/single_phase/repart_parquet"

def verify_model():
    MODEL_PATH = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl"

    print(f"🔍 Loading Multi-Output Model from {MODEL_PATH}...")

    try:
        brain = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}.")
        return    

    # Find a sample file
    files = glob.glob(f"{DATA_DIR}/*.parquet")
    if not files:
        print(f"❌ Error: No parquet files found in {DATA_DIR}")
        return
        
    df = pd.read_parquet(files[0])
    
    # Pre-process exactly as we did in training
    base_features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'wg_qty', 'fl_qty','ip_qty']
    for col in base_features:
        if col not in df.columns: 
            df[col] = 0

    fls = df['fl_qty'].replace(0, 1.0)
    wgs = df['wg_qty'].replace(0, 1.0)

    df['fl_congestion'] = (df['ipug_qty'] + df['flug_qty']) / fls
    df['wg_crowding'] = (df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']) / wgs

    df['sorties_avail'] = df['paa'] * df['ute']
    df['pilot_to_sortie'] = df['total_pilots'] / df['sorties_avail']

    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)
    
    df = df.replace([np.inf, -np.inf], 0)

    features = [
        'paa', 'ute',
        'exp_ratio', 'ip_ratio', 'fl_congestion',
        'wg_crowding', 'sorties_avail', 'pilot_to_sortie', 'ip_to_stud_ratio',
    ]
    
    targets = ['wg_monthly', 'fl_monthly', 'ip_monthly', 
               'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly']
    
    X = df[features].fillna(0)
    
    print("🧠 Generating matrix predictions...")

    preds = brain.predict(X)
    
    print("\n📊 --- REALITY CHECK METRICS (Multi-Output) ---")
    print(f"{'Target':<20} | {'Mean Value':<12} | {'MAE (Error)':<12} | {'RMSE':<12}")
    print("-" * 65)
    
    for i, target in enumerate(targets):
        if target not in df.columns: continue
            
        y_true = df[target]
        y_pred = preds[:, i] 
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"{target:<20} | {y_true.mean():<12.2f} | ± {mae:<10.2f} | {rmse:<12.2f}")

if __name__ == "__main__":
    verify_model()