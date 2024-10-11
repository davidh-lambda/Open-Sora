import concurrent.futures
import subprocess
import pandas as pd
import sys

def get_gpu_info(node_name):
    try:
        result_processes = subprocess.run(
            ["ssh", node_name, "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"],
            capture_output=True,
            text=True,
            check=True
        )
        num_processes = int(result_processes.stdout.strip())
        
        result_power = subprocess.run(
            ["ssh", node_name, "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        power_draws = [float(p.strip()) for p in result_power.stdout.splitlines()]
        mean_power = sum(power_draws) / len(power_draws) if power_draws else 0.0
        
        return node_name, num_processes, mean_power
    except subprocess.CalledProcessError as e:
        return node_name, "Failed", "Failed"

def main(hostfile):
    with open(hostfile, 'r') as file:
        nodes = [line.strip() for line in file if line.strip()]
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_gpu_info, node) for node in nodes]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
    df = pd.DataFrame(results, columns=["Node", "GPU Processes", "Mean Power Consumption (W)"])

    # Calculate mean values for GPU Processes and Mean Power Consumption
    mean_gpu_processes = df["GPU Processes"].replace("Failed", float('nan')).astype(float).mean()
    mean_power_consumption = df["Mean Power Consumption (W)"].replace("Failed", float('nan')).astype(float).mean()
    #df.loc["Mean"] = ["", mean_gpu_processes, mean_power_consumption]

    # Set pandas options to display the entire DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <hostfile>")
        sys.exit(1)
    
    hostfile = sys.argv[1]
    main(hostfile)

