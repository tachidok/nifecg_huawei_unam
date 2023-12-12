# Run the model for training and testing

```
time python nifecg_classification_multichannel_metrics_CNN.py --n_output_classes 6 --channels 0 2 4 6 --sample_interval 5 --batch_size 10 --n_epochs 1 --learning_rate 1e-2 --training_folder ../../data/sorted_by_fhr/training_set/ --testing_folder ../../data/sorted_by_fhr/testing_set/ --use_device cpu --root_output_folder RESLT
```

Run using this command considering only six classes, selecting four
channels from the thirty-two total. The `--use_device` flag allows for
`cpu`, `gpu` or `npu` devices.

If you are running on the Huawei server then you first need to start
the docker server

```
docker run -it --ipc=host --device=/dev/davinci1 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog -v /home/2022_01/project/data/sorted_by_fhr/training_set:/app/data/training_set -v /home/2022_01/project/data/sorted_by_fhr/testing_set:/app/data/testing_set -v /home/2022_01/project/src/:/tmp my_nifecg_app /bin/bash
```

once in the docker container then run the commando as indicated above
for training and testing the model.

If you are working on the Huawei server then check the `run.sh`
script. This generates the docker image by compiling a new docker
image and copying the latest version of the
`nifecg_classification_multichannel_metrics_CNN.py` file, then load
the docker image to startig testing with it.
