#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:13:28 2022

@author: almalindborg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:54:52 2022

@author: almalindborg
"""


import pandas as pd
import numpy as np
from scipy.special import psi, gamma
import seaborn as sns
import os


# Sequential Bayesian Dirichlet-categorical model
# input: Mobs, matrix
class SequentialBayesianDcat:
    """Sequential Bayesian learner model with Dirichlet-categorical distribution.
    Inputs: Mobs, N x dim matrix of observed classes, one-hot encoded and ordered vertically from first to last observation.
    functions:
        compute_alphas(tau): computes the parameters of the Dirichlet distribution at each observation, for a given value of tau
        compute_surprise(tau): computes the model's surprise values (bs, pe) for a given value of tau. Calls compute_alphas."""
        
    def __init__(self,Mobs):
        self.dim = Mobs.shape[-1]
        self.Mobs = np.concatenate((np.ones((1,self.dim)), Mobs)) # add artificial row of prior observations
    
    # estimate parameters of the Dirichlet distribution
    def compute_alphas(self, tau):
        alphas = [self.Mobs[0]] # prior
        for idx in np.arange(1,len(self.Mobs)): # loop through all observations excluding prior
            # exponential filter for previous trials
            memfilt = idx - np.tile(np.arange(idx+1), (self.dim,1)).T # last observation is always fully included
            memfilt[0] = np.zeros(self.dim)  # the prior row is always fully included
            ofilt = np.exp(-memfilt/tau)*self.Mobs[:idx+1] # filter observations
     
            # compute alpha
            a = np.sum(ofilt, axis=0)
            alphas.append(a)
          
        return np.array(alphas)
    
    # compute surprise measures
    def compute_surprise(self, tau):
        alphas = self.compute_alphas(tau)
        a_ps, a_s = alphas[:-1], alphas[1:] # predictive and updated alphas

        # compute Bayesian surprise
        bs = np.log(gamma(np.sum(a_ps, axis=1))) - np.sum(np.log(gamma(a_ps)),axis=1) - np.log(gamma(np.sum(a_s,axis=1))) + np.sum(np.log(gamma(a_s)),axis=1) - np.sum((a_s-a_ps)*(psi(a_ps)-np.tile(psi(np.sum(a_ps, axis=1)), (self.dim,1)).T),axis=1)
    
        # compute prediction error
        pe = -np.sum((self.Mobs[1:])*np.log(a_ps/np.tile(np.sum(a_ps, axis=1), (self.dim,1)).T), axis=1)
        
        return bs, pe
     
def main():
    #%% Compute surprise values for all subjects and save in csv file
    taus = np.arange(1,16)
    
    df_all =  pd.read_csv('../data/roi_300-500ms.csv')
    df_all.drop(columns='N400', inplace=True) # we don't need the N400 information here
    
    # get word category from word number
    df_all['word_cat'] = df_all['word_no'].transform(lambda x: np.floor((x-1)/10))
    
    surp_all = []
    for subject, df in df_all.groupby('Subject'):
        print('Processing subject ' + str(subject))
        
        prev_cat = np.concatenate((np.array([10]), df['word_cat'].values[:-1]))
        df['cat_switch'] = (df['word_cat'] != prev_cat).astype(int)
        
        cats_onehot = pd.get_dummies(df['word_cat']).values
        subct = SequentialBayesianDcat(cats_onehot)
        
        for tau in taus:
           bs, pe = subct.compute_surprise(tau)
           tmp = pd.DataFrame(data = {'BS': bs, 'PE': pe, 'tau': tau, 'seg': df.seg})
           df_surp = pd.merge(df, tmp, on='seg')
           surp_all.append(df_surp)              
    
    surp_all = pd.concat(surp_all)
    
    output_folder = '../outputs/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    surp_all.to_csv(output_folder + 'surprise.csv')
        
    #%% Sanity check: plot BS as function of time for one subject
    
    sdat = surp_all.loc[(surp_all.tau == 10) & (surp_all.Subject == 0),]
    sns.lineplot(x='seg', y = 'BS', data = sdat.loc[sdat.seg < 200, ])
    
if __name__ == '__main__':
    main()

