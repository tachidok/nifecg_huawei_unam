# Move signals on ranges

The script `process_database_signals.py` moves the specified signals
into folders based on the ranges provided to the script.

Example:

```
mpirun -np 4 python ./process_database_signals.py --input_folder ../01_matlab_signal_generation/output_data/heart_rate_variable/ --ext csv --input_file_names nifecg --output_folder ./sorted_signals_by_mhr --ranges 70 79 80 89 90 99 --mf m
```

* Use multiple cores to speed up the task: `mpirun -np 4` uses four
  cores to perform the task
* The `--input_folder` flag indicates the folder where the original signals are
  stored
* The `--ext` flag indicates the extension of the files that will be
  moved
* The `--input_file_names` flag indicates that only files beginning
  with `nifecg` will be moved.
* The `--output_folder` flag indicates the destination folder to be
  CREATED by the script. Subfolder will be created within this folder.
* The `--ranges` is a list of ranges for the hear rates. Folders are
  created using this rates as names. In this example three folders are
  created `70_79`, `80_89` and `90_99`. The signals in the original
  folder will be sorted based on their heart rate.
* The `--mf` flag indicates whether to use the `mother (m)` or the
  `fetal (f)` heart rate for sorting the signals. In this example, the
  `mother` hear rate is used for sorting the signals.
  
# Move signals back into their original folder
Given that the signals may be sorted using different ranges, the
script `restore_signals_to_original_folders.sh` helps to move back the
signals in the `ranges` folder to the original folder.

Example:

```
./restore_signals_to_original_folder.sh ./sorted_signals_by_mhr/ ../01_matlab_signal_generation/output_data/heart_rate_variable/
```

In this case, the signals in the `./sorted_signals_by_mhr/` folder
(and all its subfolders) are moved back into the
`../01_matlab_signal_generation/output_data/heart_rate_variable/`
folder.
