import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Prepare Data
# path = '/Users/clairebieber/air_force_scratchpad/15af/absorption_model/outputs/research_data.csv'
path = '/Users/clairebieber/air_force_scratchpad/15af/research_data_test.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.strip()

# Standardized inputs - removed ip_ratio redundancy to avoid collinearity issues
inputs = ['paa', 'ute', 'exp_ratio', 'ip_qty', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty']
X = df[inputs]

outcomes = [
    'wg_monthly', 'fl_monthly', 'ip_monthly', 
    'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly', 
    'wg_red_monthly', 'fl_red_monthly', 'ip_red_monthly'
]

# 2. Setup Regressor
model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["exp", "inv(x) = 1/x"],
    model_selection="best",
    loss="loss(prediction, target) = (prediction - target)^2",
)

results = {}

for outcome in outcomes:
    print(f"\n--- Processing: {outcome} ---")
    
    # Split data to validate against unseen scenarios
    X_train, X_test, y_train, y_test = train_test_split(X, df[outcome], test_size=0.2, random_state=42)
    
    model.fit(X_train.values, y_train.values, variable_names=inputs)
    
    # Validate using the test set
    y_pred = model.predict(X_test.values)
    
    # Calculate key interpretability metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error) 
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
    
    best_row = model.get_best()
    
    results[outcome] = {
        'eq': best_row.equation,
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'complexity': best_row.complexity
    }

# 3. Enhanced Interpretability Summary
print("\n" + "="*80)
print(f"{'OUTCOME':<20} | {'R²':<6} | {'RMSE':<6} | {'REL ERR':<8} | {'EQ'}")
print("-"*80)

for out, data in results.items():
    # Interpretability Legend:
    # R² > 0.90: Excellent fit
    # Rel Err < 5%: Highly reliable for simulation
    print(f"{out:<20} | {data['r2']:<6.3f} | {data['mape']:<7.1f}% | {data['eq']}")
print("="*80)