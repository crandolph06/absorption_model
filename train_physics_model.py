import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# --- CONFIGURATION ---
FILE_PATH = 'outputs/simulation_results.parquet'  
TARGETS = ['wg_monthly', 'fl_monthly', 'ip_monthly', 'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly']

def train_physics_model():
    # 1. Load Data
    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_parquet(FILE_PATH)
    
    # 2. Feature Engineering: The "Physics" of the FHP
    # Derive the population counts
    # Formula: num_wg = total_pilots * (1 - exp_ratio)
    df['calc_num_wg'] = df['total_pilots'] * (1 - df['exp_ratio'])
    df['calc_num_exp'] = df['total_pilots'] * df['exp_ratio']
    
    # Derive the "Iron" Capacity (Supply)
    # We assume Capacity is proportional to PAA * UTE
    df['iron_capacity'] = df['paa'] * df['ute']
    
    # Derive the "Flesh" Pressure (Demand)
    # Total student events that need IPs and Sorties
    df['student_load'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    
    # Ratios that determine the "Kinks"
    # When this ratio is high, IPs are saturated
    df['ip_saturation_index'] = df['student_load'] / df['ip_qty'].replace(0, 1)
    
    # When this ratio is high, the Fleet is saturated
    df['fleet_utilization_index'] = df['student_load'] / df['iron_capacity'].replace(0, 1)

    # 3. STRATEGY A: The "Formula" Finder (Interpretable)
    # We fit a model to find the 'Cost' of each student type.
    # Logic: Available_For_WG = (K * Capacity) - (C1*MQT + C2*FLUG + C3*IPUG)
    # We train this on the 'Total Sorties' or infer it from WG rates.
    print("\n--- DERIVING FORMULAS ---")
    
    # Calculate total monthly sorties consumed by WGs
    df['total_wg_sorties'] = df['wg_monthly'] * df['calc_num_wg']
    
    # We try to predict the "Hole" in capacity
    # Target: (PAA*UTE) - WG_Sorties ~ MQT + FLUG + IPUG
    # This regression finds the "Cost" of each upgrade.
    y_demand = df['iron_capacity'] - df['total_wg_sorties']
    X_demand = df[['mqt_qty', 'flug_qty', 'ipug_qty']]
    
    lr = LinearRegression()
    lr.fit(X_demand, y_demand)
    
    print("Discovered Physics Constants:")
    print(f"  Base Overhead (Intercept): {lr.intercept_:.2f} sorties")
    print(f"  Cost per MQT:  {lr.coef_[0]:.2f} sorties/month")
    print(f"  Cost per FLUG: {lr.coef_[1]:.2f} sorties/month")
    print(f"  Cost per IPUG: {lr.coef_[2]:.2f} sorties/month")
    print("(Use these coefficients to build your Excel/SQL formulas)")

    # 4. STRATEGY B: The "Brain" (.pkl) (Predictive)
    
    print("\n--- TRAINING PREDICTIVE BRAIN ---")
    features = [
        'paa', 'ute', 'total_pilots', 'exp_ratio', 'ip_qty',
        'mqt_qty', 'flug_qty', 'ipug_qty',
        'iron_capacity', 'ip_saturation_index', 'calc_num_wg' 
    ]
    
    models = {}
    
    for target in TARGETS:
        X = df[features].fillna(0)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Gradient Boosting Regressor
        # The 'huber' loss is robust to outliers/kinks
        gbr = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            loss='huber'
        )
        gbr.fit(X_train, y_train)
        
        score = gbr.score(X_test, y_test)
        print(f"  Target: {target:12s} | R2 Score: {score:.4f}")
        
        models[target] = gbr

    # 5. Save the Brain
    with open('pilot_training_model.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("\nModel saved to 'pilot_training_model.pkl'")
    
    return models

if __name__ == "__main__":
    # Ensure you have 'pyarrow' installed for pandas read_parquet
    # pip install pyarrow pandas scikit-learn
    train_physics_model()