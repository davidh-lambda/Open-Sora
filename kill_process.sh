#!/bin/bash

# Check if the input file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi


input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "Error: Input file '$input_file' does not exist."
  exit 1
fi

while IFS= read -r hostname || [ -n "$hostname" ]; do
  if [ -n "$hostname" ]; then
    echo "Sending 'sudo pkill -f python.*train\.py' to $hostname"
    ssh "$hostname" "sudo pkill -f python.*train\.py" &
  fi
done < "$input_file"
