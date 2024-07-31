import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Define the path to the outputs directory
outputs_dir = 'outputs'

# Create a regex pattern to extract the step counter and experiment number from the directory names
pattern = re.compile(r'global_step(\d+)')
exp_pattern = re.compile(r'outputs/(\d+)-STDiT3-XL-2')

# Create a list to store the filtered checkpoints
filtered_checkpoints = []

# Walk through the directory and gather information
for root, dirs, files in os.walk(outputs_dir):
    for dir_name in dirs:
        exp_match = exp_pattern.search(root)
        step_match = pattern.search(dir_name)
        if exp_match and step_match:
            print(exp_match)
            experiment_number = int(exp_match.group(1))
            step_counter = int(step_match.group(1))
            if experiment_number >= 180 and experiment_number not in [211, 215, 235]:
                dir_path = os.path.join(root, dir_name)
                print(dir_path)
                creation_time = os.path.getctime(dir_path)
                filtered_checkpoints.append((experiment_number, step_counter, dir_path, creation_time))

# Convert the list to a DataFrame
filtered_df = pd.DataFrame(filtered_checkpoints, columns=['experiment_number', 'step_counter', 'dir_path', 'creation_time'])

# Sort the DataFrame by step_counter and creation_time
filtered_df.sort_values(by=['step_counter', 'creation_time'], inplace=True)

# Find the latest checkpoint for each step counter
latest_filtered_checkpoints = filtered_df.groupby('step_counter').first().reset_index()

# Calculate the time elapsed between consecutive checkpoints
latest_filtered_checkpoints['time_elapsed'] = latest_filtered_checkpoints['creation_time'].diff()

# Convert the time_elapsed from seconds to a more readable format (hours)
latest_filtered_checkpoints['time_elapsed_hours'] = latest_filtered_checkpoints['time_elapsed'] / 3600
print(latest_filtered_checkpoints.to_string())

# Plot the time elapsed between consecutive checkpoints
plt.figure(figsize=(10, 6))
plt.plot(latest_filtered_checkpoints['step_counter'][2:], latest_filtered_checkpoints['time_elapsed_hours'][2:], marker='o')
plt.xlabel('Step Counter')
plt.ylabel('Time Elapsed (hours)')
plt.title('Time Elapsed Between Consecutive Checkpoints (Experiment Number >= 180)')
plt.grid(True)

# Save the plot to a file
filtered_plot_filename = 'time_elapsed_between_checkpoints_filtered.png'
plt.savefig(filtered_plot_filename)

# Save the latest_filtered_checkpoints dataframe to a CSV for inspection
latest_filtered_checkpoints.to_csv('latest_filtered_checkpoints.csv', index=False)

print(f"Plot saved as {filtered_plot_filename}")
print("Filtered checkpoints data saved as latest_filtered_checkpoints.csv")
