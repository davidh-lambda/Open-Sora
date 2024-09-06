import os
import io
import sys
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr
from opensora.datasets.read_video import read_video
from tqdm import tqdm
import traceback


def process_file(file_path):
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Capture output and error messages
    #with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
    try:
        read_video(file_path, start_pts=0, end_pts=16)
    except Exception as e:
        # Write exception details to stderr
        sys.stderr.write(f"Error processing file {file_path}: {str(e)}\n")
        sys.stderr.write(f"Stack trace:\n{traceback.format_exc()}\n")
        return  # Exit to avoid logging stdout/stderr captures in case of exception

    
    # Check if there's anything in stderr or stdout to log
    if stdout_capture.getvalue() or stderr_capture.getvalue():
        with open('errors.log', 'a') as error_file:
            error_file.write(f"Processing file: {file_path}\n")
            error_file.write(f"Captured stdout:\n{stdout_capture.getvalue()}\n")
            error_file.write(f"Captured stderr:\n{stderr_capture.getvalue()}\n")
            error_file.write("\n")  # Add a newline for separation between file entries

def find_mp4_files(root_folder):
    mp4_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(subdir, file))
    return mp4_files

def main():
    root_folder = sys.argv[-1]

    # Find all .mp4 files
    mp4_files = find_mp4_files(root_folder)

    # Process files in parallel using 204 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=204) as executor:
        for _ in tqdm(executor.map(process_file, mp4_files), total=len(mp4_files)):
            pass


if __name__ == '__main__':
    main()

