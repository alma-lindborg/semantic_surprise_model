#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:00:25 2022

@author: almalindborg
"""

import pandas as pd
import glob
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import mne


def get_trial_seq(word_no):
    trldef = word_no.to_numpy()
    trl_cat = np.floor((trldef-1)/10)
    
    trl_seq = np.ones(trldef.shape)
    for i, trl in enumerate(trl_cat[:-1]):
        if trl_cat[i+1] == trl:
            trl_seq[i+1] += trl_seq[i]
            
    return trl_seq

def read_eeg_data(path_to_dir):
    print('loading EEG data...')
    flist = list(glob.glob(path_to_dir + '/*.csv'))
    flist.sort()
    dflist = [pd.read_csv(fn) for fn in flist] 
    df_eeg = pd.concat(dflist)
    print('...done.')
    return df_eeg

#%% Condition ERP analysis over ROI
df_eeg = read_eeg_data('../data/roi_time_resolved')

# get latency time stamps from column names
lats = np.array([float(pnt) for pnt in list(df_eeg.columns[:-4])])

# compute the sequence position for each trial and add info to 
df_eeg['trl_in_seq'] = df_eeg.groupby('Subject')['word_no'].transform(lambda x: get_trial_seq(x))
df_eeg.reset_index(inplace=True, drop=True)

# find trial index of the standard and deviant trials
dev_idcs = df_eeg.loc[df_eeg.trl_in_seq == 1].index
std_idcs = dev_idcs[1:]-1

# store in data frame
df_eeg.loc['Condition'] = np.nan
df_eeg.loc[dev_idcs,'Condition'] = 'deviant'
df_eeg.loc[std_idcs,'Condition'] = 'standard'

# remove bad segments
df_eeg = df_eeg.loc[df_eeg.badseg == 0]
df_eeg.reset_index(inplace=True, drop=True)

# drop trials which are not standards or deviant and compute subject ERPs
df_conds = df_eeg.dropna()
subj_erps = df_conds.groupby(['Subject', 'Condition']).mean().reset_index()

# Stats on condition average N400 (300-500ms)
timewin = lats[(lats > 0.3) & (lats < 0.5)].astype(str)
subj_erps['mean_N400'] = subj_erps[timewin].mean(axis=1)
stat, pval = ttest_rel(subj_erps.loc[subj_erps.Condition=='deviant', 'mean_N400'], subj_erps.loc[subj_erps.Condition=='standard', 'mean_N400'])
print('p-val for deviant vs. standard t-test: %.6f' % pval)

#%% Plot condition ERPs
# grand-average for plotting
ga = subj_erps.groupby('Condition').mean().iloc[:,1:-5].transpose()
ga.index = ga.index.astype(float)

fig, ax = plt.subplots(1,1)
stl, = ax.plot(ga.index, ga.standard, label = 'standard')
devl, = ax.plot(ga.index, ga.deviant, label = 'deviant')
plt.legend(handles=[stl, devl])
plt.xlabel('latency (s.)')

ax.hlines(xmin = 0.3, xmax=0.5, y=-2, color='black')
plt.scatter(x=0.4+np.array([-0.01, 0.01]), y=-2.2*np.ones(2), marker='*', color='black') # annotate with stars
plt.savefig('../plots/standard_deviant_roi.pdf')
plt.show()

#%% TOPOPLOTS
# load channel data
df_eeg = read_eeg_data('../data/channel_300-500ms')

# Topoplot of standard and deviant difference
df_eeg['trl_in_seq'] = df_eeg.groupby('Subject')['word_no'].transform(lambda x: get_trial_seq(x))
df_eeg.reset_index(inplace=True)

dev_idcs = df_eeg.loc[df_eeg.trl_in_seq == 1].index
std_idcs = dev_idcs[1:]-1

diff_ga = df_eeg.iloc[dev_idcs].mean() - df_eeg.iloc[std_idcs].mean()

# create montage and MNE data array
montage = mne.channels.make_standard_montage('standard_1005')
ch_list = list(diff_ga.index[2:-5])
info = mne.create_info(ch_names = ch_list, sfreq  = 128.0, ch_types = 'eeg')

evoked = mne.EvokedArray(data=diff_ga.loc[ch_list].values.reshape(-1, 1).astype(np.float64), info=info) 
evoked.set_montage(montage)

# mark ROI
roi = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4']
mask_pms = dict(marker='*', markerfacecolor='r', markeredgecolor='r',
        linewidth=0, markersize=8)
nr_idx = np.zeros(len(ch_list))
for channel in roi: nr_idx[ch_list.index(channel)] = 1
nr_idx = nr_idx.astype(bool).reshape(-1, 1)

# make plot
fig, ax = plt.subplots(1,2)
evoked.plot_topomap(times=[0], axes=ax[0], mask=nr_idx, mask_params = mask_pms, scalings = 1,  cbar_fmt="%.1f",  colorbar = True, show=False)
ax[0].set_title('deviant - standard')
plt.savefig('../plots/dev_minus_std_topo_with_roi.pdf')
plt.show()
