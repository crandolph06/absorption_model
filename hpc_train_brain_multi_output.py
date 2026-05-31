
import pandas as pd
import numpy as np
import joblib
import glob
import os

from src.syllabi import SORTIE_SLOTS_MQT, SORTIE_SLOTS_FLUG, SORTIE_SLOTS_IPUG

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

INPUT_DIR = "outputs/single_phase/repart_parquet"
SAMPLE_FRAC = 0.10 
RANDOM_SEED = 42

def train_hpc_multi_brain():
    print(f"🚀 Starting Multi-Output HPC Brain Training...")
    files = glob.glob(os.path.join(INPUT_DIR, "part.*.parquet"))
    
    if not files:
        print(f"❌ No parquet files found in {INPUT_DIR}")
        return

    # LOAD & SAMPLE DATA
    mini_batches = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        if SAMPLE_FRAC < 1.0:
            df_chunk = df_chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        mini_batches.append(df_chunk)
    
    df = pd.concat(mini_batches, ignore_index=True)
    print(f"📊 Dataset loaded: {len(df):,} rows")

    # FEATURE & TARGET SELECTION
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

    if "incomplete_mqt_students_mean" in df.columns:
        df["deferred_mqt_students"] = df["incomplete_mqt_students_mean"]
        df["deferred_flug_students"] = df["incomplete_flug_students_mean"]
        df["deferred_ipug_students"] = df["incomplete_ipug_students_mean"]
    else:
        df["deferred_mqt_students"] = df["deferred_mqt_lines_mean"] / SORTIE_SLOTS_MQT
        df["deferred_flug_students"] = df["deferred_flug_lines_mean"] / SORTIE_SLOTS_FLUG
        df["deferred_ipug_students"] = df["deferred_ipug_lines_mean"] / SORTIE_SLOTS_IPUG

    df = df.replace([np.inf, -np.inf], 0)

    features = [
        'paa', 'ute', 'exp_ratio', 'ip_ratio', 'fl_congestion',
        'wg_crowding', 'sorties_avail', 'pilot_to_sortie', 'ip_to_stud_ratio',
    ]
    
    targets = [
        'wg_monthly', 'fl_monthly', 'ip_monthly', 
        'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly',
        'remaining_mqt_syllabi_mean', 'remaining_flug_syllabi_mean', 'remaining_ipug_syllabi_mean',
        'remaining_mqt_syllabi_sorties_only_mean', 'remaining_flug_syllabi_sorties_only_mean', 'remaining_ipug_syllabi_sorties_only_mean'
    ]

    X = df[features].fillna(0)
    Y = df[targets].fillna(0)

    # 3. SPLIT
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

    OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl"

    # DEFINE MODEL
    brain = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',       # Critical for learning 'hinges' and floors
        solver='adam',           # Standard optimizer for large datasets
        alpha=0.001,             # L2 Regularization (prevents the 'wiggles')
        batch_size=1024,         # Helps with the 50M row memory load
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,     # Stops once it stops improving on the test set
        random_state=42,
        verbose=True
    )
    mlp_model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', brain)
    ])

    # TRAIN
    print("🧠 Training MLP Regressor model...")
    mlp_model.fit(X_train, Y_train)

    # EVALUATE
    print("MLP Model trained! Evaluating performance...")
    y_pred = mlp_model.predict(X_test)

    # SCORE
    score = r2_score(Y_test, y_pred)
    print(f"✅ Training Complete. Overall R² Score: {score:.4f}")

    # SAVE
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    
    joblib.dump(mlp_model, OUTPUT_MODEL)
    print(f"💾 MLP brain saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    train_hpc_multi_brain()