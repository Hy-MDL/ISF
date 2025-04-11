# timegan-pytorch
This repository holds the code for the reimplementation of TimeGAN 

## Getting Started


### Directory Hierarchy
```bash
data/                         # the folder dataset
  ├ data_preprocessing.py     # the data preprocessing functions
  ├ sliding_windows_...csv    # sliding window Data
  └ energy_numeric.csv        # Origin Data(numeric)
metrics/                      # the folder holding the metric functions for evaluating the model
  ├ dataset.py                # the dataset class for feature predicting and one-step ahead predicting
  ├ general_rnn.py            # the model for fitting the dataset during TSTR evaluation
  ├ metric_utils.py           # the main function for evaluating TSTR
  └ visualization.py          # PCA and t-SNE implementation for time series taken from the original repo
models/                       # the code for the model
output/                       # the output of the model
main.py                       # the main code for training and evaluating TSTR of the model
requirements.txt              # requirements for running code
run.sh                        # the bash script for running model
visualization.ipynb           # jupyter notebook for running visualization of original and synthetic data
```
###
In terminal -> run.sh (If you need to change some statement, you can change directly)