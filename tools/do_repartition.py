
import dask.dataframe as dd
import glob
import os
INPUT_PATH = "outputs/single_phase/parquet/*.parquet"
OUTPUT_DIR = "outputs/single_phase/repart_parquet"

def repartition_data():
    n_files = len(glob.glob(INPUT_PATH))
    print(f"🔍 Reading {n_files} files from {INPUT_PATH}...")
    
    # Load all files lazily
    df = dd.read_parquet(INPUT_PATH)
    
    print(f"🔄 Re-partitioning into 50 files...")
    # This redistribute the data into 50 equal-ish chunks
    df = df.repartition(npartitions=50)
    
    print(f"💾 Writing to {OUTPUT_DIR}...")
    df.to_parquet(OUTPUT_DIR, engine='pyarrow', write_index=False)
    
    print("🎉 Done! Check the 'repart_parquet' folder.")

if __name__ == "__main__":
    repartition_data()

