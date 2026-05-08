import pandas as pd
import pickle
from src.models import AgingRate

class CAFModel:
    def __init__(self, model_path='../pilot_training_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Error: 'pilot_training_model.pkl' not found.")
            self.models = None

    def predict(self, paa, ute, total_pilots, exp_ratio, ip_qty, mqt_qty, flug_qty, ipug_qty):
        """
        Predicts monthly rates for WG, FL, and IP.
        
        Args:
            mqt_qty (int): Number of MQT students (Impact: Low impact on WG rate)
            flug_qty (int): Number of FLUG students (Impact: Moderate impact)
            ipug_qty (int): Number of IPUG students (Impact: High impact - drains IPs)
        """
        if not self.models:
            return None

        # 1. ORGANIZE INPUTS
        # We assume '0' for missing values to be safe
        data = {
            'paa': [paa],
            'ute': [ute],
            'total_pilots': [total_pilots],
            'exp_ratio': [exp_ratio],
            'ip_qty': [ip_qty],
            'mqt_qty': [mqt_qty],
            'flug_qty': [flug_qty],
            'ipug_qty': [ipug_qty]
        }
        df = pd.DataFrame(data)

        # 2. CALCULATE PHYSICS FEATURES (The "Hint" Columns)
        # Supply
        df['iron_capacity'] = df['paa'] * df['ute']
        
        # Demand (Weighted Pressure)
        df['student_load'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
        
        # Constraints
        df['calc_num_wg'] = df['total_pilots'] * (1 - df['exp_ratio'])
        
        # Saturation Index (IP Pressure)
        # Avoid division by zero
        df['ip_saturation_index'] = df.apply(
            lambda row: row['student_load'] / row['ip_qty'] if row['ip_qty'] > 0 else 999, axis=1
        )

        # 3. DEFINE FEATURE ORDER (Must match training exactly)
        features = [
            'paa', 'ute', 'total_pilots', 'exp_ratio', 'ip_qty',
            'mqt_qty', 'flug_qty', 'ipug_qty',           
            'iron_capacity', 'ip_saturation_index', 'calc_num_wg'
        ]

        # 4. RUN PREDICTION
        results = {}
        for target in ['wg_monthly', 'fl_monthly', 'ip_monthly', 'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly']:
            val = self.models[target].predict(df[features])[0]
            results[target] = max(0.0, round(val, 2)) # Clip negative values

        monthly_rates = AgingRate(
            mqt_phase=4.0,
            wg_phase=results['wg_monthly'],
            fl_phase=results['fl_monthly'],
            ip_phase=results['ip_monthly'],
            mqt_blue_phase=4.0,
            wg_blue_phase=results['wg_blue_monthly'],
            fl_blue_phase=results['fl_blue_monthly'],
            ip_blue_phase=results['ip_blue_monthly']
        )
            
        return monthly_rates

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Initialize the predictor
    predictor = CAFModel()

    # Scenario 1: High IPUG (Should hurt WG rate)
    print("\n--- Scenario: High IPUG ---")
    res1 = predictor.predict(
        paa=18, ute=16, total_pilots=40, exp_ratio=0.4, ip_qty=6, 
        mqt_qty=0, 
        flug_qty=0, 
        ipug_qty=4  # High IPUG
    )
    print(f"WG Rate: {res1['wg_monthly']} (Expected: Lower)")

    # Scenario 2: High MQT (Should NOT hurt WG rate as much)
    print("\n--- Scenario: High MQT ---")
    res2 = predictor.predict(
        paa=18, ute=16, total_pilots=40, exp_ratio=0.4, ip_qty=6, 
        mqt_qty=4,  # High MQT
        flug_qty=0, 
        ipug_qty=0
    )
    print(f"WG Rate: {res2['wg_monthly']} (Expected: Higher)")