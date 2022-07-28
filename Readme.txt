This repository contains the code and data to reproduce the results from Lindborg, Musiolek, Ostwald, Rabovsky (2022), “Semantic Surprise predicts the N400 brain potential” (Submitted).

The repository is organised as follows.

%%%% Code %%%%

The code is organised in 4 scripts.
1. erp_analysis.py
Runs the ERP analysis described in the main text and outputs the figures into the ./plots subfolder.
2. compute_model_surprise.py
Computes the semantic surprise of the Bayesian sequential learner on the stimulus sequences. Saves the model surprise (necessary for subsequent analyses) to ./outputs/surprise.csv
3. ./R_code_mixed_model/n400_mixed_models.Rmd
R markdown document containing the mixed model analysis. Estimates the best value of the forgetting parameter tau (necessary for subsequent analyses) and saves to ./outputs/best_tau.csv
4. encoding_model_analysis.py
Computes the temporally and spatially resolved models of the N400 as predicted by semantic surprise.

%%%% DATA %%%%

Summarised data is available in three forms:
1. Time-averaged N400 over ROI
2. Time-resolved signal over ROI
3. Channel data for average N400 time window

All data files contain the following columns:

- seg: segment (trial) number in experimental session (1-3000)
- word_no: ID of the stimulus presented in the trial (1-100). Stimulus IDs can be found in the materials/Stimuli.csv file. Stimulus numbers are grouped such that word 1-10 belong to one category, 11-20 another, etc.
- Subject: Subject number (0-39).
- badseg: 1 if the trial is marked as bad (to be excluded from further analysis), 0 otherwise.

In addition, the files contain data columns stated below.

%%% Time-averaged N400 (300-500 ms) over ROI:
% roi_300-500ms.csv
Contains the average N400 potential over the ROI used in the main analyses.
electrodes: 'F3', 'F1', 'Fz', 'F2', 'F4', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4'

% central_parietal_roi_300-500ms.csv
Contains the average N400 potential over the ROI used in the supplementary material.
electrodes: 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'

Both files have the following data column:
- N400: Average voltage over ROI

%%% Time-resolved data over the ROI
ROI as in main text.
Data files are found in the roi_time_resolved subfolder (#.csv where # is the subject number).

Data columns:
-0.09375, …, 0.78906: 114 columns representing the signal at each time sample (relative to stimulus onset)

%%% Channel data for average N400
Time window: 300-500 ms.
Data files are found in the channel_300-500ms subfolder (#.csv where # is the subject number).

Data columns: 
AF7, … O2: channel labels.



