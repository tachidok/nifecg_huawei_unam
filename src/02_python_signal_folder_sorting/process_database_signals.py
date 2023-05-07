#!/usr/bin/env python

import os
import sys
import shutil
import argparse

from mpi4py import MPI

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def find_files(input_folder, file_extensions, initial_file_names):

    '''
    Scan for files that start with initial_file_name and end with the
    file extension in a folder (and all its subfolders).

    Returns:
    - The total number of found files that meet the criteria
    - The list of filenames that meet the criteria

    '''
    
    # A list with the full names of the files with the given extensions    
    file_list = []
    # A counter for the number of found files
    file_count = 0

    for root, dirs, files in os.walk(input_folder):
        for ifile in files:
            # For each file check whether its meets the criteria
            for file_extension in file_extensions:
                for initial_file_name in initial_file_names:
                    if ifile.endswith(file_extension) and ifile.startswith(initial_file_name):
                        file_list.append(os.path.join(root, ifile))
                        file_count += 1
    
    return file_list, file_count

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def move_file_into_subfolder(item, ranges):

    '''
    Move a file in the corresponding folder based on its range
    '''
    
    # Get the signal path
    signal_path = item.split("/")
    
    # Get the signal name
    signal_name = signal_path[-1]

    # Get the signal type
    signal_type = signal_name.split(".")
    
    # Get signal features
    signal_features = signal_type[2].split("_")

    # Based on the signal range, move it into the corresponding folder
    print(signal_features)

    HERE HERE HERE HERE
        
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def main():
    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_cores = comm.Get_size()
    
    # Create the parser to deal with the arguments
    parser = argparse.ArgumentParser("Find the files with a given extension and a given heart rate range and create subfolder within the specified ranges")
    
    # Set the positional arguments
    parser.add_argument("--input_folder", type=str, help="The input folder to search for signals", required=True)
    parser.add_argument("--ext", type=str, nargs='+', help="The list of extensions for the files to look for", required=True, choices=['csv', 'png', 'jpeg', 'bmp', 'txt', 'jpg'])
    parser.add_argument("--input_file_names", type=str, nargs='+', help="The initial filenames of the signals", required=True)
    parser.add_argument("--output_folder", type=str, help="The output folder to sorted signals", required=True)
    parser.add_argument("--ranges", type=int, nargs='+', help="A list of sorted ranges to sort the signals in subfolders", required=True)
    
    # parse args
    args = parser.parse_args()

    print("\n")
    
    # -------------------------------------------------------------
    # Validate input arguments
    # -------------------------------------------------------------
    
    # Only master core check if the output folder already exists
    if rank == 0 and os.path.exists(args.output_folder):
        parser.error('The output folder already exists')
    
    if len(args.ranges) % 2 != 0:
        parser.error('The number of input numbers in ranges must be even')
    
    # Store the ranges
    ranges = [tuple(args.ranges[i:i+2]) for i in range(0, len(args.ranges), 2)]

    # Only the master core creates the subfolders
    if rank == 0:
        # create a folder for each pair in ranges
        for irange in ranges:
            range_str = '_'.join(map(str, irange))
            range_path = os.path.join(args.output_folder, range_str)
            os.makedirs(range_path, exist_ok=True)
    
        print(f"{rank}/{n_cores}::Created {len(ranges)} folders for the ranges: {ranges}")
    
    # All cores wair for the master core to create the subfolders
    comm.Barrier()
        
    # --------------------------------------------------------------
    # Information
    # --------------------------------------------------------------
    print(f"{rank}/{n_cores}::Searching for files beginning with '{args.input_file_names}' and with extensions '{args.ext}' in '{args.input_folder}' folder and all subfolders ...\n")
    print(f"{rank}/{n_cores}::Ranges for subfolders are:'{ranges}'\n")
    print(f"{rank}/{n_cores}::Scanning ...\n")
    
    # Scan for files of a given type and return the filenames list
    found_files, n_found_files = find_files(args.input_folder, args.ext, args.input_file_names)
    
    print(f"{rank}/{n_cores}::Found [{n_found_files}] files with the specified features")
    
    # Create the list of image features
    ###  HERE MOVE THE FILES INTO THE SPECIFIED FOLDER
    
    # --------------------------------------------------------------
    # Move files into folders
    # --------------------------------------------------------------
    if n_found_files > 0:
        counter = rank
        
        while counter < n_found_files:
            item = found_files[counter]
            print(f"{rank}/{n_cores}::{counter}/{n_found_files}: Processing {item}")
            
            move_file_into_subfolder(item, ranges)
                        
            counter+=n_cores
        
    else:
        print(f"{rank}/{n_cores}::No found files with the specified features")
    
    print(f"\n{rank}/{n_cores}::[Done]\n")

    # Count the number of files on each folder and check whether it
    # adds up to the total number of signals
    
    
    # Finish MPI
    MPI.Finalize()
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
'''
Usage:
   ./process_database_signals.py --input_folder input_folder --ext extensions of files --output_folder output_folder --ranges --ranges
'''
if __name__ == '__main__':
    # Run the main function
    main()
