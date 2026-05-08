
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
from sklearn.neural_network import MLPRegressor

INPUT_DIR = "outputs/single_phase/repart_parquet"
# OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_hybrid.pkl" 
OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl" 
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
    base_features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'wg_qty', 'fl_qty','ip_qty']
    for col in base_features:
        if col not in df.columns: 
            df[col] = 0
    ips = df['ip_qty'].replace(0, 0.5)
    fls = df['fl_qty'].replace(0, 1.0)
    wgs = df['wg_qty'].replace(0, 1.0)

    df['mqt_load'] = df['mqt_qty'] / ips
    df['flug_load'] = df['flug_qty'] / ips
    df['ipug_load'] = df['ipug_qty'] / ips
    
    df['fl_congestion'] = (df['ipug_qty'] + df['flug_qty']) / fls
    df['wg_crowding'] = (df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']) / wgs

    df['sorties_avail'] = df['paa'] * df['ute']
    df['pilot_to_sortie'] = df['total_pilots'] / df['sorties_avail']

    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)
    
    df = df.replace([np.inf, -np.inf], 0)
    features = [
        'exp_ratio', 'ip_ratio', 'fl_congestion',
        'wg_crowding', 'sorties_avail', 'pilot_to_sortie', 'ip_to_stud_ratio'
    ]
    
    targets = [
        'wg_monthly', 'fl_monthly', 'ip_monthly', 
        'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly'
    ]

    X = df[features].fillna(0)
    Y = df[targets].fillna(0)

    # 3. SPLIT
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

    # # 4. DEFINE HYBRID MODEL
    # base_model = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('ridge', Ridge(alpha=1.0))
    # ])
    # linear_model = MultiOutputRegressor(base_model)
    # print("🧠 Training Linear model...")
    # linear_model.fit(X_train, Y_train)

    # wg_coefs = linear_model.estimators_[0].named_steps['ridge'].coef_
    # print(f"WG Ridge Coefs: {wg_coefs}")    
    # fl_coefs = linear_model.estimators_[1].named_steps['ridge'].coef_
    # print(f"FL Ridge Coefs: {fl_coefs}")
    # ip_coefs = linear_model.estimators_[2].named_steps['ridge'].coef_
    # print(f"IP Ridge Coefs: {ip_coefs}")

    # Y_train_pred_lin = linear_model.predict(X_train)

    # print("🧮 Calculating Residuals...")
    # residuals = Y_train - Y_train_pred_lin

    # booster_model = MultiOutputRegressor(
    #     HistGradientBoostingRegressor(
    #     max_iter=150,
    #     learning_rate=0.03,     
    #     max_leaf_nodes=10,      
    #     min_samples_leaf=1000,   
    #     l2_regularization=50.0,  
    #     random_state=RANDOM_SEED
    #     )
    # )
    # print("🧠 Training Booster model...")
    # booster_model.fit(X_train, residuals)

    # # 5. EVALUATE
    # print("Models trained! Evaluating performance...")
    # y_pred_lin_test = linear_model.predict(X_test)
    # y_pred_res_test = booster_model.predict(X_test)
    # y_pred_total = y_pred_lin_test + y_pred_res_test

    brain = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',       # Critical for learning 'hinges' and floors
        solver='adam',           # Standard optimizer for large datasets
        alpha=0.001,             # L2 Regularization (prevents the 'wiggles')
        batch_size=1024,         # Helps with the 50M row memory load
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,     # Stops once it stops improving on the test set
        random_state=42
    )
    mlp_model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', brain)
    ])

    print("🧠 Training MLP Regressor model...")
    mlp_model.fit(X_train, Y_train)
    print("MLP Model trained! Evaluating performance...")
    y_pred = mlp_model.predict(X_test)


    score = r2_score(Y_test, y_pred)
    # score = r2_score(Y_test, y_pred_total)
    print(f"✅ Training Complete. Overall R² Score: {score:.4f}")

    # 6. SAVE
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    # hybrid_brain = {
    #     'linear': linear_model,
    #     'booster': booster_model
    # }
    # joblib.dump(hybrid_brain, OUTPUT_MODEL)
    # print(f"💾 Hybrid brain saved to {OUTPUT_MODEL}")
    MLP_brain = mlp_model
    joblib.dump(mlp_model, OUTPUT_MODEL)
    print(f"💾 MLP brain saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    train_hpc_multi_brain()