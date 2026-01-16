import pandas as pd
import numpy as np
import os

def diagnose_lookup(df_path: str, params: dict, priority_vars: list):
    """
    Loads the parquet file and runs a verbose, step-by-step 
    simulation of the lookup logic.
    """
    print(f"\n{'='*80}")
    print(f"🕵️‍♂️ DIAGNOSTIC RUN")
    print(f"📁 File:   {df_path}")
    print(f"🎯 Target: {params}")
    print(f"🔢 Priority Order: {priority_vars}")
    print(f"{'='*80}\n")

    # --- 1. Load Data ---
    if not os.path.exists(df_path):
        print(f"❌ ERROR: File not found at {df_path}")
        return

    try:
        df = pd.read_parquet(df_path)
        print(f"✅ Loaded DataFrame with {len(df)} rows.")
        # Print dtypes to check for float/int mismatches
        print(f"   Dtypes Check -> PAA: {df['paa'].dtype}, UTE: {df['ute'].dtype}, IP_QTY: {df.get('ip_qty', pd.Series()).dtype}")
    except Exception as e:
        print(f"❌ ERROR Loading Parquet: {e}")
        return

    # --- 2. Environmental Lock (Exact Match) ---
    target_paa = params.get('paa')
    target_ute = params.get('ute')
    
    # Check if columns exist
    if 'paa' not in df.columns or 'ute' not in df.columns:
        print("❌ ERROR: 'paa' or 'ute' columns missing from DataFrame.")
        return

    mask = (df['paa'] == target_paa) & (df['ute'] == target_ute)
    subset = df[mask]
    
    print(f"\n🔹 STEP 1: PAA ({target_paa}) & UTE ({target_ute}) Lock")
    print(f"   Rows Remaining: {len(subset)}")
    
    if len(subset) == 0:
        print("   ⚠️  WARNING: No exact match for PAA/UTE. Your code falls back to full dataset.")
        # Replicating your fallback logic
        mask = np.ones(len(df), dtype=bool)
        subset = df
    else:
        # Show range of what's available in this bucket
        print(f"   Stats for this PAA/UTE bucket:")
        for col in ['ip_qty', 'exp_ratio', 'total_pilots']:
            if col in subset.columns:
                print(f"     - {col}: Min {subset[col].min():.4f} | Max {subset[col].max():.4f}")

    # --- 3. Sequential Drill Down ---
    current_mask = mask
    
    for i, var in enumerate(priority_vars):
        print(f"\n🔹 STEP 2.{i+1}: Filter by '{var}'")
        
        target_val = params.get(var, 0)
        
        # Check if column exists
        if var not in df.columns:
            print(f"   ⚠️  Column '{var}' not found in DataFrame! Skipping.")
            continue
            
        # Get values of currently surviving rows
        valid_values = df.loc[current_mask, var].values
        
        if len(valid_values) == 0:
            print("   ⚠️  No rows left to filter!")
            break

        # Calculate differences for ALL survivors to show what the code "sees"
        diffs = np.abs(valid_values - target_val)
        min_diff = np.min(diffs)
        
        print(f"   Target Value: {target_val}")
        print(f"   Minimum Difference Available: {min_diff:.6f}")
        
        # Identify who survives this round
        # Using your exact epsilon logic: (min_diff + 0.00001)
        is_closest_mask = np.abs(df[var] - target_val) <= (min_diff + 0.00001)
        new_mask = current_mask & is_closest_mask
        
        survivors = df[new_mask]
        num_dropped = len(df[current_mask]) - len(survivors)
        
        print(f"   Rows Kept: {len(survivors)} (Dropped {num_dropped})")
        
        # CRITICAL: Show a sample of what was kept vs dropped to spot the logic error
        if len(survivors) > 0:
            print(f"   👀 Sample Survivor (Index {survivors.index[0]}):")
            print(f"      {var} = {survivors.iloc[0][var]} (Diff: {abs(survivors.iloc[0][var] - target_val):.4f})")
            print(f"      Sortie Rates: WG={survivors.iloc[0].get('wg_monthly', 'N/A')}, IP={survivors.iloc[0].get('ip_monthly', 'N/A')}")
        
        current_mask = new_mask

    # --- 4. Final Result ---
    # We skip the student matrix distance for this test as it requires external matrix objects, 
    # but we can grab the first surviving row (which is what usually happens if dists are equal)
    
    if not any(current_mask):
        print("\n❌ RESULT: No rows survived the filters.")
    else:
        final_idx = np.where(current_mask)[0][0]
        final_row = df.iloc[final_idx]
        
        print(f"\n{'='*80}")
        print("✅ FINAL LOOKUP RESULT")
        print(f"{'='*80}")
        print(final_row[['exp_ratio', 'ip_qty', 'total_pilots', 'wg_monthly', 'fl_monthly', 'ip_monthly']].to_frame().T)
        print("\nIf 'wg_monthly' is ~5-6 and 'ip_monthly' is ~6, you pulled a healthy row.")

# --- RUN CONFIGURATION ---
if __name__ == "__main__":
    
    # 📝 UPDATE THIS PATH to your real file
    FILE_PATH = 'outputs/simulation_results.parquet' 

    # 📝 UPDATE THESE PARAMS to the specific failure case you see in the chart
    # (The values from your prompt: 26% exp ratio, but getting ~6 sorties)
    DEBUG_PARAMS = {
        'paa': 24,           # Update if your chart uses different PAA
        'ute': 10,           # Update if your chart uses different UTE
        'exp_ratio': 0.23,   # The target ratio
        'ip_qty': 3,        # The estimated IP count for that ratio/manning
        'total_pilots': 39   # The estimated total pilots
    }

    # 📝 THE PRIORITY ORDER
    # This matches your original code snippet.
    DEBUG_PRIORITY = ['ip_qty', 'exp_ratio', 'total_pilots']

    diagnose_lookup(FILE_PATH, DEBUG_PARAMS, DEBUG_PRIORITY)