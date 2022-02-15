# Source code Details

## data folder

After data_exploration.py and AUTOML.ipynb, necessary new *.csv files have been build in data folder.

## NMF.py, PMY.py, CF_KNN.py CF_DL.py AUTOML.py

These files contain origin versions of models without hyperparameters tuning. Just run those files and you will get primitive predications made by raw models.

## Best_Parameters&Performance.md

This file contains best parameters after tuning. If you want to skip hyperparameters tuning and directly have the final prediction results, you can just follow the parameter list in this file and plug them into raw models mentioned above.

## Hyperparameters_Tuning folder

Take NMF as example.

NMF.py in this folder is a modified version for parameters_choosing.

NMF_search_space.json is a json script to define search space.

NMF_config.yml is a config file to define trial files, tuning algorithm, max trial number.

a reference operation steps are as below:

1. Update the model codes. Simply Modify NMF.py to get the hyperparameter set from NNI and report the final results to NNI.
2. Define the search space. We define the value range of hyperparameters that we want to tune in NMF_search_space.json.

3. Config the experiment and specify the key information of the experiment, such as the trial files, tuning algorithm, max trial number, etc.

4. Launch the experiment from the command line by nnictl create --config NMF.yml.

5. View the Experiment by opening the Web UI url in browser and we can view overview, trials detail and experiments management.











