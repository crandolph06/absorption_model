import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Prepare Data
path = '/Users/clairebieber/air_force_scratchpad/15af/absorption_model/outputs/research_data.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.strip()

# Standardized inputs 
sum_students = ['mqt_qty', 'flug_qty', 'ipug_qty']

df['ip_to_stud_ratio'] = df['ip_qty'] / df[sum_students].sum(axis=1).clip(lower=1)
df['ip_ratio'] = df['ip_qty'] / df['total_pilots']
df['ac_per_pilot'] = df['paa'] / df['total_pilots']
df['ute_per_pilot'] = df['ute'] / df['total_pilots']
df['ip_load'] = (df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']) / (df['ip_qty'])
df['exp_ratio_sq'] = df['exp_ratio']**2
df['upgrade_pct'] = (df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']) / (df['total_pilots'])
df['max_capacity'] = (df['paa'] * df['ute'])

engineered_inputs = ['ip_to_stud_ratio'] + ['ip_ratio'] + ['ac_per_pilot'] + ['ute_per_pilot'] + ['exp_ratio'] + ['ip_load'] + ['exp_ratio_sq'] + ['max_capacity'] + ['upgrade_pct']
# raw_inputs = ['paa', 'ute', 'exp_ratio', 'ip_qty', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty']
raw_inputs = ['total_pilots']

X = df[engineered_inputs + raw_inputs]
input_names = X.columns.tolist()

# outcomes = [
#     'wg_monthly', 'fl_monthly', 'ip_monthly', 
#     'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly', 
#     'wg_red_monthly', 'fl_red_monthly', 'ip_red_monthly'
# ]
outcomes = ['wg_monthly']

# 2. Setup Regressor
model = PySRRegressor(
    niterations=1000,             
    maxsize=35,                  
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["inv", "square", "sqrt"], 
    constraints={'/': (-1, 1)},  
    extra_sympy_mappings={"inv": lambda x: 1/x},
    batching=True,
    batch_size=1024,
    model_selection="best",
    loss="loss(prediction, target) = log(cosh(prediction - target))",
)

results = {}
log_trans = True
last_lcosh_loss = float('inf')

def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))

for outcome in outcomes:
    print(f"\n--- Processing: {outcome} ---")
    
    # Split data to validate against unseen scenarios
    X_train, X_test, y_train, y_test = train_test_split(X, df[outcome], test_size=0.2, random_state=42)
    
    if log_trans:
        y_train_log = np.log1p(y_train)
        model.fit(X_train.values, y_train_log.values, variable_names=input_names)
        y_pred_log = model.predict(X_test.values)
        y_pred = np.expm1(y_pred_log)

    else:
        model.fit(X_train.values, y_train.values, variable_names=input_names)
        y_pred = model.predict(X_test.values)
    
    best_row = model.get_best()

    # Calculate key interpretability metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error) 
    epsilon = .1
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

    log_cosh_val = log_cosh_loss(y_test, y_pred)
    log_cosh_imp_pct = ((last_lcosh_loss - log_cosh_val) / last_lcosh_loss) * 100 if last_lcosh_loss != 0 else 0
    
    results[outcome] = {
        'eq': best_row.equation,
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'l-cosh': log_cosh_val,
        'l-cosh imp': log_cosh_imp_pct,
        'complexity': best_row.complexity
    }

    last_lcosh_loss = log_cosh_val

# 3. Enhanced Interpretability Summary
print("\n" + "="*80)
print(f"{'OUTCOME':<20} | {'R²':<6} | {'MAPE':<8} | {'L-COSH IMP.':<8} | {'EQ'}")
print("-"*80)

for out, data in results.items():
    # Interpretability Legend:
    # R² > 0.90: Excellent fit
    # Rel Err < 5%: Highly reliable for simulation
    print(f"{out:<20} | {data['r2']:<6.3f} | {data['mape']:<7.1f}% | {data['l-cosh imp']:<7.1f}% | {data['eq']}")
print("="*80)