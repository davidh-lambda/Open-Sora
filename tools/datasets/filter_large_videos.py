import sys
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Function to get the size of a file
def get_file_size(path):
    return os.path.getsize(path) if os.path.isfile(path) else None

# Load the CSV file
file_path = sys.argv[-2]
size = int(sys.argv[-1])
assert len(sys.argv) == 3
df = pd.read_csv(file_path)

# Enable tqdm to monitor progress
tqdm.pandas()

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your CPU cores
    df['file_size'] = list(tqdm(executor.map(get_file_size, df['path']), total=len(df)))

# Convert 50 MB to bytes
size_threshold = size * 1024 * 1024  # 50 MB in bytes

# Drop rows where file size is 50 MB or more
df_filtered = df[df['file_size'] < size_threshold]

# Save the filtered DataFrame back to the original CSV file
file_path = file_path.replace(".csv", f"_le{size}M.csv")
df_filtered.to_csv(file_path.replace(".csv", f"_le{size}M.csv"), index=False)
print(f"Saved filtered data to {file_path}.")

