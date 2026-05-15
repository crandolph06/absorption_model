
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import glob
import os

INPUT_PATH = "outputs/single_phase/parquet/*.parquet"
OUTPUT_DIR = "outputs/single_phase/repart_parquet"

def repartition_data():
    # 1. Start a strict memory-managed cluster
    # 4 workers at 14GB each = 56GB total (Leaving 8GB for OS overhead to prevent OOM)
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='14GB')
    client = Client(cluster)
    
    n_files = len(glob.glob(INPUT_PATH))
    print(f"🔍 Reading {n_files} files from {INPUT_PATH}...")
    
    # Load all files lazily
    df = dd.read_parquet(INPUT_PATH)
    
    print(f"🔄 Re-partitioning into 50 files...")
    # This redistributes the data into 50 equal-ish chunks
    df = df.repartition(npartitions=50)
    
    print(f"💾 Writing to {OUTPUT_DIR}...")
    df.to_parquet(OUTPUT_DIR, engine='pyarrow', write_index=False)
    
    print("🎉 Done! Check the 'repart_parquet' folder.")
    
    # Shut down the cluster gracefully
    client.close()
    cluster.close()

if __name__ == "__main__":
    repartition_data()
