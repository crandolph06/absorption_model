import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
path = "outputs/simulation_results.parquet"  
print(f"📂 Loading {path}...")
df = pd.read_parquet(path)

# Prepare Inputs (X) and Target (y)
features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
target = 'wg_monthly'

# Fill missing cols
for col in features:
    if col not in df.columns: df[col] = 0

X = df[features].values
y = df[target].values

# ==============================================================================
# 2. COMPETITOR A: The "Black Box" (Random Forest)
# ==============================================================================
# Random Forest makes NO assumptions about physics. It just follows the data.
print("\n🤖 Training Random Forest (The Benchmark)...")
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_model.fit(X, y)
y_pred_rf = rf_model.predict(X)
r2_rf = r2_score(y, y_pred_rf)

print(f"   Benchmark R² Score: {r2_rf:.4f}")
if r2_rf > 0.90:
    print("   ✅ VERDICT: Data is PREDICTABLE. The Physics Equation is the problem.")
else:
    print("   ❌ VERDICT: Data is NOISY. Even a flexible AI couldn't predict it well.")

# ==============================================================================
# 3. COMPETITOR B: The "Simplified" Physics Equation
# ==============================================================================
# I removed the exponential constraint and the hard cutoffs to see if a simpler
# linear-style physics model fits better.
def simple_physics(X_in, a, b, c, d, e, f):
    paa, ute, ratio, pilots, mqt, flug, ipug, ips = X_in.T
    
    # 1. Base Capacity (PAA * UTE / Pilots)
    # We let 'a' scale the pilots (effective pilot count)
    wg_share = pilots * (1 - ratio)
    base_rate = (paa * ute) / np.maximum(wg_share, 1.0)
    
    # 2. Linear Penalty for Students (No complex taxes, just a straight subtraction)
    # Rate = Base - (Student_Load / Pilots)
    penalty = (b*mqt + c*flug + d*ipug) / np.maximum(pilots, 1.0)
    
    # 3. Simple IP Bonus/Penalty
    # Instead of exponential, just a linear factor
    ip_factor = e * (ips / np.maximum(pilots, 1.0))
    
    return a * base_rate - penalty + ip_factor + f

print("\n📐 Training Simplified Physics Equation...")
try:
    popt, _ = curve_fit(simple_physics, X, y, p0=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    y_pred_phys = simple_physics(X, *popt)
    r2_phys = r2_score(y, y_pred_phys)
    print(f"   Physics R² Score: {r2_phys:.4f}")
except Exception as e:
    print(f"   Physics Fit Failed: {e}")

# ==============================================================================
# 4. SHOW THE DIFFERENCE
# ==============================================================================
print("\n🔍 Comparison on 5 Random Rows:")
indices = np.random.choice(len(df), 5)
for i in indices:
    print(f"   Actual: {y[i]:.2f} | Random Forest: {y_pred_rf[i]:.2f} | Physics: {y_pred_phys[i]:.2f}")