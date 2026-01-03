import pandas as pd

df = pd.read_csv('outputs/research_data.csv')

df.to_parquet('simulation_results.parquet', engine='pyarrow', compression='brotli')

print("Conversion complete. Check the file size of simulation_results.parquet!")