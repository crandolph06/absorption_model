import pandas as pd
import os

PATH="outputs/long_term"
FILE_NAME="final_long_term_batch"

df = pd.read_parquet(os.path.join(PATH, f'{FILE_NAME}.parquet'))

df.to_csv(os.path.join(PATH, f'{FILE_NAME}.csv'))

print(f"Conversion complete. Check the file size of {FILE_NAME}.csv")