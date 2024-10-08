#!/bin/bash
HOME=$(dirname $(dirname "$(realpath "$0")"))
echo $HOME

# Define the list of files and nodes
files=($PATH_SPLITS/part_*)

nodes=($NODES_IDX)

# Check if the number of files and nodes match
if [ ${#files[@]} -ne ${#nodes[@]} ]; then
  echo ${#files[@]}
  echo ${#nodes[@]}
  echo "Error: Number of files and nodes do not match."
  exit 1
fi

# Iterate over the files and nodes arrays
for i in "${!files[@]}"; do
  file=${files[$i]}
  node=${nodes[$i]}

  ssh ${NODES_NAME}${node} bash -c "cd $HOME;
dir=\"${HOME}/clip_folders/$file\"
mkdir -p \"${HOME}/ffmpeg_errors\"
echo \"Processing file: \$dir\"
parallel --progress -j+0 'ffmpeg -v error -i {} -f null - 2>{}.err' < \$dir
" &

done
