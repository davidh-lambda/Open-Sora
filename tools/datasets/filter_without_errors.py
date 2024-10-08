import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

def check_error_file(row):
    # Construct the path to the .err file
    err_file = f"{row['path']}.err"
    
    # Check if the .err file exists and is empty
    if not os.path.exists(err_file):
        return row
    elif os.path.exists(err_file) and os.path.getsize(err_file) == 0:
        return row
    return None

def filter_csv(input_csv):
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(input_csv)
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(df), desc="Processing files")
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submit tasks for parallel execution
        futures = {executor.submit(check_error_file, row): index for index, row in df.iterrows()}
        
        # Collect results as they complete
        filtered_rows = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                filtered_rows.append(result)
            progress_bar.update(1)
    
    # Close the progress bar
    progress_bar.close()
    
    # Create a new DataFrame from the filtered rows
    filtered_df = pd.DataFrame(filtered_rows, columns=df.columns)
    
    # Generate the output file name
    output_csv = input_csv.replace('.csv', '_withouterror.csv')
    
    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved as: {output_csv}")

if __name__ == "__main__":
    # Replace 'yourfile.csv' with the actual CSV file you want to process
    import sys
    input_csv = sys.argv[-1]
    filter_csv(input_csv)

