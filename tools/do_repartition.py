import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import glob
import os

INPUT_PATH = "outputs/single_phase/parquet/*.parquet"
OUTPUT_DIR = "outputs/single_phase/repart_parquet"

def repartition_data():
    # 1. Aggressive Memory Management (CRITICAL)
    # Forces Dask to spill to disk at 70% instead of crashing
    dask.config.set({
        'distributed.worker.memory.target': 0.60,
        'distributed.worker.memory.spill': 0.70,
        'distributed.worker.memory.pause': 0.80,
        'distributed.worker.memory.terminate': 0.95
    })
    
    # 2. One massive worker
    # 50GB gives it plenty of room to breathe inside your 64GB Slurm limit
    # 8 threads let it read/write files much faster
    cluster = LocalCluster(n_workers=1, threads_per_worker=8, memory_limit='50GB')
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