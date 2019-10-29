### Output files for PKU evaluation are at
https://www.dropbox.com/sh/jk1mf3prgr3lje1/AACEyqq2Ob84lR6pVf-nKnQ5a?dl=0

## Demo 
https://youtu.be/r-TXSpMnvhY

# Experiments


## 1. Run the experiments
Just run the scripts in the `bin` folder.

`train_dahlia_rgb.sh`: Train I3D on Dahlia RGB

`train_dahlia_skeletons.sh`: Train LSTM on Dahlia Skeletons

`train_pku_rgb.sh`: Train I3D on PKU-MMD RGB

`train_pku_skeletons`: Train LSTM on PKU-MMD Skeletons

## 2. Arguments

* `dataset_name`: `pku-mmd` or `dahlia`
* `model`: `i3d` or `lstm`
* `num_epochs`: Number of epochs
* `batch_size`: Batch size for training
* `eval_batch_size`: Batch size for evaluation
* `experiment_name`: Name of the experiment 
* `log_dir`: Directory to keep all the result in
* `num_gpus`: Number of GPUs

### 2.2 Dahlia
* `dahlia_train_path`: Path to the training file
* `dahlia_validation_path`: Path to the validation file
* `dahlia_test_path`: Path to the testing file
* `dahlia_skeletons_dir`: Directory to get the skeletos from
* `dahlia_modality`: `rgb` or `skeletons`

### 2.3 PKU-MMD
* `pku_mmd_split`: `cross_view` or `cross_subject`
* `pku_mmd_rgb_dir`: Directory to get the RGB frames from
* `pku_mmd_skeletons_dir`: Directory to get the skeletons from
* `pku_mmd_labels_dir`: Directory to get the label files from
* `pku_mmd_splits_dir'`: Directory to get the split file from
* `pku_mmd_modality`: `rgb` or `skeletons`

### 2.4 LSTM
* `dropout`: Dropout
* `time_steps`: Number of time steps
* `num_neurons`: Number of neurons
* `data_size`: Size of the input feature vector

