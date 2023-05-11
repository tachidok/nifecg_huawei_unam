#!/bin/bash

# Set the source and destination directories
#src_dir="./sorted_signals_by_mhr"
#dst_dir="../01_matlab_signal_generation/output_data/heart_rate_variable/"

src_dir="$1"
dst_dir="$2"

# Move all files in the source directory to the destination directory
find "$src_dir" -type f -exec mv -i '{}' "$dst_dir" \;
