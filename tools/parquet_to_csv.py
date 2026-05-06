import pandas as pd
import os

PATH="outputs/hpc"
FILE_NAME="batch_0450"

df = pd.read_parquet(os.path.join(PATH, f'{FILE_NAME}.parquet'))

df.to_csv(os.path.join(PATH, f'{FILE_NAME}.csv'))

print(f"Conversion complete. Check the file size of {FILE_NAME}.csv")
