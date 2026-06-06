
import pandas as pd
import numpy as np
import joblib
import glob
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

INPUT_DIR = "outputs/single_phase/repart_parquet"
LOW_BATCH_GLOB = "outputs/single_phase/parquet/batch_low_*.parquet"
OUTPUT_MODEL = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl"
SAMPLE_FRAC = 0.10
LOW_EXP_THRESHOLD = 0.10
LOW_EXP_SAMPLE_FRAC = 1.0
RANDOM_SEED = 42

TARGETS = [
    "wg_monthly", "fl_monthly", "ip_monthly",
    "wg_blue_monthly", "fl_blue_monthly", "ip_blue_monthly",
    "mqt_sim_monthly", "wg_sim_monthly", "fl_sim_monthly", "ip_sim_monthly",
    "remaining_mqt_syllabi_mean", "remaining_flug_syllabi_mean", "remaining_ipug_syllabi_mean",
    "remaining_mqt_syllabi_sorties_only_mean", "remaining_flug_syllabi_sorties_only_mean",
    "remaining_ipug_syllabi_sorties_only_mean",
]


def model_output_count(model):
    preds = model.predict(np.zeros((1, 9)))
    return preds.shape[1] if preds.ndim > 1 else 1


def require_target_columns(df):
    missing = [col for col in TARGETS if col not in df.columns]
    if missing:
        print(f"❌ Training data is missing {len(missing)} required target column(s):")
        for col in missing:
            print(f"   - {col}")
        print("   Re-run repartition from current sweep parquet, then retrain.")
        sys.exit(1)


def sample_chunk(df_chunk):
    """Keep all low-exp rows; sample high-exp rows at SAMPLE_FRAC."""
    low = df_chunk[df_chunk["exp_ratio"] <= LOW_EXP_THRESHOLD]
    high = df_chunk[df_chunk["exp_ratio"] > LOW_EXP_THRESHOLD]

    if LOW_EXP_SAMPLE_FRAC < 1.0:
        low = low.sample(frac=LOW_EXP_SAMPLE_FRAC, random_state=RANDOM_SEED)
    elif LOW_EXP_SAMPLE_FRAC > 1.0 and len(low) > 0:
        low = low.sample(frac=LOW_EXP_SAMPLE_FRAC, replace=True, random_state=RANDOM_SEED)

    if len(high) > 0 and SAMPLE_FRAC < 1.0:
        high = high.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)

    return pd.concat([low, high], ignore_index=True)


def train_hpc_multi_brain():
    print("🚀 Starting Multi-Output HPC Brain Training...")
    print(f"   Script expects {len(TARGETS)} outputs (includes 4 sim monthly targets)")
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "part.*.parquet")))
    low_files = sorted(glob.glob(LOW_BATCH_GLOB))

    if not files and not low_files:
        print(f"❌ No parquet files found in {INPUT_DIR} or {LOW_BATCH_GLOB}")
        return

    # LOAD & SAMPLE DATA
    mini_batches = []
    for f in files:
        mini_batches.append(sample_chunk(pd.read_parquet(f)))
    for f in low_files:
        mini_batches.append(pd.read_parquet(f))

    df = pd.concat(mini_batches, ignore_index=True)
    require_target_columns(df)
    low_exp_count = (df["exp_ratio"] <= LOW_EXP_THRESHOLD).sum()
    print(f"📊 Dataset loaded: {len(df):,} rows")
    print(f"   exp_ratio <= {LOW_EXP_THRESHOLD}: {low_exp_count:,} rows")

    # FEATURE & TARGET SELECTION
    base_features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'wg_qty', 'fl_qty','ip_qty']
    for col in base_features:
        if col not in df.columns: 
            df[col] = 0

    df['fl_congestion'] = (df['ipug_qty'] + df['flug_qty']) / df['fl_qty']
    df['wg_crowding'] = (df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']) / df['wg_qty']

    df['sorties_avail'] = df['paa'] * df['ute']
    df['pilot_to_sortie'] = df['total_pilots'] / df['sorties_avail']

    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)

    df = df.replace([np.inf, -np.inf], 0)

    features = [
        'paa', 'ute', 'exp_ratio', 'ip_ratio', 'fl_congestion',
        'wg_crowding', 'sorties_avail', 'pilot_to_sortie', 'ip_to_stud_ratio',
    ]
    
    X = df[features].fillna(0)
    Y = df[TARGETS].fillna(0)
    print(f"   Training matrix: X={X.shape}, Y={Y.shape}")

    # 3. SPLIT
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

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

    low_test = X_test["exp_ratio"] <= LOW_EXP_THRESHOLD
    if low_test.any():
        low_score = r2_score(Y_test.loc[low_test], y_pred[low_test], multioutput="uniform_average")
        print(f"   R² (exp_ratio <= {LOW_EXP_THRESHOLD}): {low_score:.4f} ({low_test.sum():,} test rows)")

    # SAVE
    n_outputs = model_output_count(mlp_model)
    print(f"   Saved model output count: {n_outputs}")
    if n_outputs != len(TARGETS):
        print(f"❌ Refusing to save: expected {len(TARGETS)} outputs, got {n_outputs}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    joblib.dump(mlp_model, OUTPUT_MODEL)
    print(f"💾 MLP brain saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    train_hpc_multi_brain()