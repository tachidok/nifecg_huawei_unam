#!/usr/bin/env python

import os
import sys
import shutil
import argparse
import random

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
def move_file_into_subfolder(file_path, folder_path, ranges, use_mother_or_fetal_hr):

    '''
    Move a file in the corresponding folder based on its range
    '''
    
    # Get the signal path
    signal_path = file_path.split("/")
    
    # Get the signal name
    signal_name = signal_path[-1]

    # Get the signal type
    signal_type = signal_name.split(".")
    
    # Get signal features
    signal_features = signal_type[2].split("_")

    # Based on the signal range, move it into the corresponding folder
    print(signal_features)
    
    index = 0
    if (use_mother_or_fetal_hr == 'm'):
        index = 3

    if (use_mother_or_fetal_hr == 'f'):
        index = 5

    # Get the heart rate
    hr = int(signal_features[index])
    # Get the corresponding range for the hr
    for irange in ranges:
        if (hr >= irange[0] and hr <= irange[1]):
            # Move the file into the corresponding folder
            range_str = '_'.join(map(str, irange))
            range_path = os.path.join(folder_path, range_str)
            
            file_name = os.path.basename(file_path)
            new_file_path = os.path.join(range_path, file_name)

            #shutil.copy(file_path, new_file_path)
            shutil.move(file_path, new_file_path)

            # The file was moved
            return True

    # There is no corresponding range for the file
    return False
        
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
    parser.add_argument("--ranges", type=int, nargs='+', help="A list of sorted ranges to sort the signals into subfolders", required=True)
    parser.add_argument("--mf", type=str, help="Use mother (m) or fetal (f) heart rate to sort signals into subfolders", required=True, choices=['m', 'f'])
    parser.add_argument("--percentage_for_training_set", type=float, help="The percentage of files for the training set", required=True)
    
    # parse args
    args = parser.parse_args()

    print("\n")
    
    # -------------------------------------------------------------
    # Validate input arguments
    # -------------------------------------------------------------

    # Only master core check if the output folder already exists
    if rank == 0 and os.path.exists(args.output_folder):
        parser.error('The output folder already exists')
    
    output_training_folder_name = os.path.join(args.output_folder, "training_set")
    output_testing_folder_name = os.path.join(args.output_folder, "testing_set")
    # Only master core check if the output folder already exists
    if rank == 0 and os.path.exists(output_training_folder_name):
        parser.error('The output training folder already exists')

    if rank == 0 and os.path.exists(output_testing_folder_name):
        parser.error('The output testing folder already exists')
        
    if len(args.ranges) % 2 != 0:
        parser.error('The number of input numbers in ranges must be even')
    
    # Store the ranges
    ranges = [tuple(args.ranges[i:i+2]) for i in range(0, len(args.ranges), 2)]
    
    # Only the master core creates the subfolders
    if rank == 0:
        # create a folder for each pair in ranges
        for irange in ranges:
            range_str = '_'.join(map(str, irange))

            # Create folder in the training folder
            range_path = os.path.join(output_training_folder_name, range_str)
            os.makedirs(range_path, exist_ok=True)

            # Create folder in the testing folder
            range_path = os.path.join(output_testing_folder_name, range_str)
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
    n_moved_files = 0
    n_no_moved_files = 0

    # Generate a list with the indices of the files
    percentage_for_training_set = args.percentage_for_training_set
    n_files_for_training_set = int(percentage_for_training_set * n_found_files)
    n_files_for_testing_set = n_found_files - n_files_for_training_set

    # Generate a list with the indices of the files
    files_indices = list(range(n_found_files))
    
    # Shuffle the input list randomly
    random.shuffle(files_indices)

    # Generate the list with the indices for the training files
    training_files_indices = files_indices[:n_files_for_training_set]
    testing_files_indices = files_indices[n_files_for_training_set:]

    if rank == 0:
        print("The training files indices")
        print(training_files_indices)

        print("The testing files indices")
        print(testing_files_indices)
        
    # --------------------------------------------------------------
    # Move files into folders
    # --------------------------------------------------------------
    if n_found_files > 0:

        # **********************************
        # Move the files for training
        # **********************************
        counter = rank
        
        while counter < len(training_files_indices):

            file_index = training_files_indices[counter]
            item = found_files[file_index]
            print(f"{rank}/{n_cores}::{counter}/{n_files_for_training_set}: To training set {item}")
            
            moved = move_file_into_subfolder(item, output_training_folder_name, ranges, args.mf)
            if moved:
                n_moved_files+=1
            else:
                n_no_moved_files+=1

            counter+=n_cores
        
        # **********************************
        # Move the files for testing
        # **********************************
        counter = rank
        
        while counter < len(testing_files_indices):
            file_index = testing_files_indices[counter]
            item = found_files[file_index]
            print(f"{rank}/{n_cores}::{counter}/{n_files_for_testing_set}: To test set {item}")
            
            moved = move_file_into_subfolder(item, output_testing_folder_name, ranges, args.mf)
            if moved:
                n_moved_files+=1
            else:
                n_no_moved_files+=1
                        
            counter+=n_cores
        
    else:
        print(f"{rank}/{n_cores}::No found files with the specified features")

    print(f"{rank}/{n_cores}::The number of moved files is {n_moved_files}\n")
    print(f"{rank}/{n_cores}::The number of NO moved files is {n_no_moved_files}\n")
    
    print(f"\n{rank}/{n_cores}::[Done]\n")

    # Count the number of files on each folder and check whether it
    # adds up to the total number of signals
    
    
    # Finish MPI
    MPI.Finalize()
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Run the main function
    main()
