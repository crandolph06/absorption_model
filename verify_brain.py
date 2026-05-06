
import pandas as pd

import numpy as np

import joblib

import glob

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



MODEL_PATH = "outputs/single_phase/brains/hpc_sortie_brain_lite.pkl"

DATA_DIR = "outputs/single_phase/repart_parquet"



def verify_model():

    print(f"🔍 Loading Model from {MODEL_PATH}...")

    try:

        brain = joblib.load(MODEL_PATH)

    except FileNotFoundError:

        print(f"❌ Error: Model file not found at {MODEL_PATH}. Is training finished?")

        return

    

    # Find a sample file

    files = glob.glob(f"{DATA_DIR}/*.parquet")

    if not files:

        print(f"❌ Error: No parquet files found in {DATA_DIR}")

        return

        

    test_file = files[0]

    print(f"📂 Loading Test Data for Validation: {test_file}")

    

    df = pd.read_parquet(test_file)

    

    # Pre-process exactly as we did in training

    base_features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']

    for col in base_features:

        if col not in df.columns: df[col] = 0

            

    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']

    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)

    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)

    df = df.replace([np.inf, -np.inf], 0)

    

    X = df[base_features + ['ip_ratio', 'ip_to_stud_ratio']]

    

    print("\n📊 --- REALITY CHECK METRICS ---")

    print(f"{'Target':<20} | {'Mean Value':<12} | {'MAE (Error)':<12} | {'RMSE':<12}")

    print("-" * 65)

    

    for target, model in brain.items():

        if target not in df.columns: continue

            

        y_true = df[target]

        y_pred = model.predict(X)

        

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        

        print(f"{target:<20} | {y_true.mean():<12.2f} | ± {mae:<10.2f} | {rmse:<12.2f}")

        

        # Leakage warning

        if mae < (y_true.mean() * 0.001) and mae != 0:

            print(f"   ⚠️ WARNING: {target} has near-zero error. Inspect for data leakage!")



if __name__ == "__main__":

    verify_model()

