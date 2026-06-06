
import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "outputs/single_phase/repart_parquet"

TARGETS = [
    "wg_monthly", "fl_monthly", "ip_monthly",
    "wg_blue_monthly", "fl_blue_monthly", "ip_blue_monthly",
    "mqt_sim_monthly", "wg_sim_monthly", "fl_sim_monthly", "ip_sim_monthly",
    "remaining_mqt_syllabi_mean", "remaining_flug_syllabi_mean", "remaining_ipug_syllabi_mean",
    "remaining_mqt_syllabi_sorties_only_mean", "remaining_flug_syllabi_sorties_only_mean",
    "remaining_ipug_syllabi_sorties_only_mean",
]


def verify_model():
    MODEL_PATH = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl"

    print(f"🔍 Loading Multi-Output Model from {MODEL_PATH}...")

    try:
        brain = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}.")
        return

    if os.path.exists(MODEL_PATH):
        mtime = pd.Timestamp(os.path.getmtime(MODEL_PATH), unit="s")
        print(f"   Model file modified: {mtime}")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "part.*.parquet")))
    if not files:
        print(f"❌ Error: No parquet files found in {DATA_DIR}")
        return

    df = pd.read_parquet(files[0])
    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        print(f"⚠️  Verify parquet missing {len(missing_targets)} target column(s):")
        for col in missing_targets:
            print(f"   - {col}")
        print("   Repartition from current sweep data to score sim outputs against ground truth.")

    # Pre-process exactly as we did in training
    base_features = ["paa", "ute", "exp_ratio", "total_pilots", "mqt_qty", "flug_qty", "ipug_qty", "wg_qty", "fl_qty", "ip_qty"]
    for col in base_features:
        if col not in df.columns:
            df[col] = 0

    df["fl_congestion"] = (df["ipug_qty"] + df["flug_qty"]) / df["fl_qty"]
    df["wg_crowding"] = (df["mqt_qty"] + df["flug_qty"] + df["ipug_qty"]) / df["wg_qty"]

    df["sorties_avail"] = df["paa"] * df["ute"]
    df["pilot_to_sortie"] = df["total_pilots"] / df["sorties_avail"]

    df["total_students"] = df["mqt_qty"] + df["flug_qty"] + df["ipug_qty"]
    df["ip_ratio"] = df["ip_qty"] / df["total_pilots"].replace(0, 1)
    df["ip_to_stud_ratio"] = df["ip_qty"] / df["total_students"].replace(0, 0.1)

    df = df.replace([np.inf, -np.inf], 0)

    features = [
        "paa", "ute",
        "exp_ratio", "ip_ratio", "fl_congestion",
        "wg_crowding", "sorties_avail", "pilot_to_sortie", "ip_to_stud_ratio",
    ]

    X = df[features].fillna(0)

    print("🧠 Generating matrix predictions...")
    preds = brain.predict(X)
    n_outputs = preds.shape[1] if preds.ndim > 1 else 1
    print(f"   Model outputs: {n_outputs}  |  Expected: {len(TARGETS)}")

    if n_outputs != len(TARGETS):
        print("❌ Output count mismatch — retrain with current hpc_train_brain_multi_output.py")
        return

    print("\n📊 --- REALITY CHECK METRICS (Multi-Output) ---")
    print(f"{'Target':<40} | {'Mean True':<12} | {'Mean Pred':<12} | {'MAE':<12} | {'RMSE':<12}")
    print("-" * 95)

    for i, target in enumerate(TARGETS):
        y_pred = preds[:, i]
        pred_mean = float(np.mean(y_pred))

        if target not in df.columns:
            print(f"{target:<40} | {'N/A':<12} | {pred_mean:<12.2f} | {'N/A':<12} | {'N/A':<12}")
            continue

        y_true = df[target]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(
            f"{target:<40} | {y_true.mean():<12.2f} | {pred_mean:<12.2f} | "
            f"± {mae:<10.2f} | {rmse:<12.2f}"
        )

if __name__ == "__main__":
    verify_model()