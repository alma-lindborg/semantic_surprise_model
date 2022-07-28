# library imports
import pandas as pd
import glob
import numpy as np
from scipy.stats import ttest_1samp, normaltest, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeCV, Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os

# helper functions 
def encoding_model(df_eeg, df_surp, rr_mod, meas):
    # merge data
    df_all = pd.merge(df_eeg, df_surp, on=['seg', 'Subject', 'word_no', 'badseg'], how='inner')
    # make sure data are correctly scaled
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    eeg_cols = df_all.iloc[:,:-9].to_numpy()
    eeg_scaled = std_scaler.fit_transform(eeg_cols)
    
    pred_col = df_all.loc[:,meas].to_numpy()
    pred_scaled = minmax_scaler.fit_transform(pred_col.reshape(-1, 1))

    fit = rr_mod.fit(pred_scaled, eeg_scaled)
    
    return fit

def stattest(coef):
    k2, pnorm = normaltest(coef)
    if pnorm > 0.05:
        statistic, pval = ttest_1samp(coef, 0)
    else:
        statistic, pval = wilcoxon(coef)
    return pval

## Variables for encoding model analysis
# best tau derived from mixed model analysis in R
best_tau = pd.read_csv('../outputs/best_taus.csv')
best_tau = best_tau.drop(columns = best_tau.columns[0]).to_dict('records')[0]

# data frame storing pre-computed surprise values for all values of memory decay (tau)
df_alltau = pd.read_csv('../outputs/surprise.csv')
df_alltau = df_alltau.iloc[:,1:]

#%% load channel EEG data
flist = list(glob.glob('../data/channel_300-500ms/*.csv'))
flist.sort()
dflist = [];
for sidx, fn in enumerate(flist):
    print('loading subject ' + str(sidx+1) + ' of ' + str(len(flist)))
    tmp = pd.read_csv(fn)
    dflist.append(tmp)
    
df_eeg = pd.concat(dflist)
df_eeg = df_eeg.loc[df_eeg.badseg == 0,]
df_eeg.dropna(inplace=True)

ch_list = list(df_eeg.columns[:-4]) # channel names   

#%% Grand-average encoder model in channel space
# fit encoder model for BS and PE with their respective best values of tau

plot_folder = '../plots/'
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

rcv = RidgeCV(alphas=np.logspace(-10,10,21)) # crossvalidated ridge regression

avgfits = []
for meas, val in best_tau.items():
    df = df_alltau.loc[df_alltau.tau == val]
    fit = encoding_model(df_eeg, df, rcv, meas)
    fw_fit = pd.DataFrame(data = {'Channel': ch_list, 'measure': meas, 'coef': fit.coef_.flatten(), 'tau': val, 'alpha': fit.alpha_})
    avgfits.append(fw_fit)

# concatenate into data frame
avgfits = pd.concat(avgfits)

# plot
montage = mne.channels.make_standard_montage('standard_1005')
for meas, group in avgfits.groupby('measure'):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)

    info = mne.create_info(ch_names = group.Channel.values.tolist(), sfreq  = 128.0, ch_types = 'eeg')
    evoked = mne.EvokedArray(data=group.coef.values.reshape(-1, 1).astype(np.float64), info=info) 
    evoked.set_montage(montage)
    evoked.plot_topomap(times=[0], axes=ax[0], scalings = 1, units = 'a.u.', cbar_fmt="%.3f", colorbar = True, show=False)
    ax[0].set_title(' ')
    
    plt.savefig(plot_folder + 'topo_encoding_grandavg_' + meas + '_best_tau.pdf')
    plt.show()
    
#%% Read in time-resolved ROI data
flist = list(glob.glob('../data/roi_time_resolved/*.csv'))
flist.sort()
dflist = [];
for sidx, fn in enumerate(flist):
    print('loading subject ' + str(sidx+1) + ' of ' + str(len(flist)))
    
    tmp = pd.read_csv(fn)
    dflist.append(tmp)
    
df_eeg = pd.concat(dflist)
df_eeg = df_eeg.loc[df_eeg.badseg == 0,]
df_eeg.dropna(inplace=True)
# get latency time stamps from column names
lats = [float(pnt) for pnt in list(df_eeg.columns[:-4])]

#%% Grand-average encoding model (to find shrinkage factor)
rcv = RidgeCV(alphas=np.logspace(-10,10,21)) # crossvalidated ridge regression

avgfits = []
for meas, val in best_tau.items():
    df = df_alltau.loc[df_alltau.tau == val]
    fit = encoding_model(df_eeg, df, rcv, meas)
    fw_fit = pd.DataFrame(data = {'Latency': lats, 'measure': meas, 'coef': fit.coef_.flatten(), 'tau': val, 'alpha': fit.alpha_})
    avgfits.append(fw_fit)
avgfits = pd.concat(avgfits).reset_index(drop=True)

sns.lineplot(x = 'Latency', y = 'coef', style = 'measure', data = avgfits).set_xlabel('Latency (s.)')
plt.savefig(plot_folder + 'encoding_time_grandavg.png')
plt.show()

#%%  Run subject-specific encoding models with the optimal alpha

# shrinkage factor
alpha = avgfits.alpha.unique()[0]
# pick alpha from global encoding model
rreg = Ridge(alpha = alpha) # no cross-validation (alpha is fixed)

allcoefs = []
for subject, subj_eeg in df_eeg.groupby('Subject'):
    for meas, val in best_tau.items():
        df = df_alltau.loc[df_alltau.tau == val]
        fit = encoding_model(subj_eeg, df, rreg, meas)
        fw_fit = pd.DataFrame(data = {'Latency': lats, 'measure': meas, 'coef': fit.coef_.flatten(), 'tau': val, 'Subject': subject})
        allcoefs.append(fw_fit)

coefs_df = pd.concat(allcoefs)   

#%% Stats on subject-specific coefficients

# Compute p-value for each measure, tau and latency and correct for multiple comparisons
coefs_df['pval'] = coefs_df.groupby(['Latency', 'measure'])['coef'].transform(lambda x: stattest(x))
coefs_df['pval'] = coefs_df.groupby(['Subject', 'measure'])['pval'].transform(lambda x: multipletests(x, method='holm')[1])
coefs_df['significant'] = coefs_df['pval'] < 0.05
coefs_df.reset_index(inplace=True, drop=True)

# save to CSV file
#coefs_df.to_csv('../outputs/time_resolved_encoding_coefs_with_stats.csv')

#%% Plot for each measure with significances

sem = 68 # plot standard error of the mean as error bands (68% confint)

for meas, group in coefs_df.groupby('measure'):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.hlines(xmin = group.Latency.min(), xmax = coefs_df.Latency.max(), y = 0, color='dimgrey', linestyle='dashed')
    sns.lineplot(x = 'Latency', y = 'coef', ci=sem, data = group, ax=ax)
   
    # find significant coefficients and mark
    sig_coefs = group.loc[group.significant == 1]
    lb = sig_coefs.groupby('Latency').coef.mean().min() - 0.01 # y coordinate for significance marks
    sig_lats = sig_coefs.Latency.unique()
    print(str(len(sig_lats))+ ' significant time points for ' + meas)
    plt.scatter(x = sig_lats, y = [lb]*len(sig_lats), marker = '.', color = 'k', alpha = 0.8, s = 6)
    
    # axis labels and origin lines
    ax.set_ylabel('a. u.')
    ax.set_xlabel('Latency (s.)')
    ax.set_xlim(group.Latency.min(), group.Latency.max())
    ax.set_ylim(lb-0.005, 0.01)
    sns.despine()
    
    plt.savefig(plot_folder + 'subject_encoding_coefs_' + meas + '_besttau.pdf')    
    plt.show()
