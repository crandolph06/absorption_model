import pandas as pd

# 1. Load your "broken" CSV
df = pd.read_csv("outputs/research_data.csv")

# 2. Reverse-engineer the total_pilots
# Since: experienced = int(total_pilots * exp_ratio) 
# And: ip_qty is a subset of experienced...
# We can find total_pilots by matching it back to your original loop logic.

def infer_total_pilots(row):
    # Your loop had [30, 35, 40]
    # We check which of those totals matches the IP and EXP RATIO math
    for t in [30, 35, 40]:
        # This is the exact math from your create_pilots function
        if int(t * row['exp_ratio']) >= row['ip_qty']:
            # We check the WG count logic too
            wg_count = t - int(t * row['exp_ratio'])
            if row['mqt_qty'] + row['flug_qty'] <= wg_count:
                return t
    return None

if 'total_pilots' not in df.columns:
    print("Repairing total_pilots column...")
    df['total_pilots'] = df.apply(infer_total_pilots, axis=1)
    
    # 3. Save the repaired version
    df.to_csv("outputs/research_data.csv", index=False)
    print("Salvage complete! You can now refresh Streamlit.")