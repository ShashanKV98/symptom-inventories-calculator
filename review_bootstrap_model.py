# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:13:01 2024

@author: vshas
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from crosswalk_symptom_functional import set_crosswalk_files,crosswalk_scores
from funcs import *

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

df = pd.read_excel("Symptom_Data.xlsx")

bsi_cols = [col for col in df.columns.values if col.startswith("BSI")]
df['bsi_recorded'] = 1*(df[bsi_cols].isnull().sum(axis=1) <= 1)

rpq_cols = [col for col in df.columns.values if col.startswith("RPQ")]
df['rpq_recorded'] = 1*(df[rpq_cols].isnull().sum(axis=1) <= 1)

df_bsi_rpq_only = df[(df['bsi_recorded'] == 1) & (df['rpq_recorded'] == 1)].reset_index().drop(columns= 'index')
rpq_cols = rpq_cols[:16]
overlap_symptoms = rpq_cols + bsi_cols
df_bsi_rpq_only = df_bsi_rpq_only[overlap_symptoms].fillna(2) 
reorder_ind = np.random.permutation(df_bsi_rpq_only.index)
df_bsi_rpq_only = df_bsi_rpq_only.reindex(index = reorder_ind).reset_index().drop(columns = ['index'])

df_linear = pd.DataFrame(index= df_bsi_rpq_only.index, columns=rpq_cols + bsi_cols)


def run_lg(train_cols,test_cols,test_size = 0.5, verbose = True):
    X = df_bsi_rpq_only[train_cols]
    y = df_bsi_rpq_only[test_cols[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    if verbose:
        print('-------------------------------------------------------')
        print('Running LG prediction models per column, 1st half...')
        print('-------------------------------------------------------')
        
    for i in range(len(test_cols)):
        
        if verbose:
            print(i,'/',len(test_cols) ,  test_cols[i])
        
        y_train = df_bsi_rpq_only.loc[X_train.index,test_cols[i]]
        y_test = df_bsi_rpq_only.loc[X_test.index,test_cols[i]]
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = np.round(regr.predict(X_test))
        # y_pred = 1 * (y_pred > 3)
        # y_test = 1 * (y_test > 3)
        df_linear.loc[X_test.index,test_cols[i]] = (y_pred == y_test).astype(int)
        # df_linear.loc[X_test.index,test_cols[i]] = abs(y_pred- y_test)
    
    X_new_train = X_test
    X_new_test = X_train
    
    if verbose:
        print('-------------------------------------------------------')
        print('Running LG prediction models per column, 2nd half...')
        print('-------------------------------------------------------')
    
    for i in range(len(test_cols)):
        
        if verbose:
            print(i,'/',len(test_cols) ,  test_cols[i])
            
        y_train = df_bsi_rpq_only.loc[X_new_train.index,test_cols[i]]
        y_test = df_bsi_rpq_only.loc[X_new_test.index,test_cols[i]]
        regr = LinearRegression()
        regr.fit(X_new_train, y_train)
        y_pred = np.round(regr.predict(X_new_test))
        # y_pred = 1 * (y_pred > 3)
        # y_test = 1 * (y_test > 3)
        df_linear.loc[X_train.index,test_cols[i]] = (y_pred == y_test).astype(int)
        # df_linear.loc[X_train.index,test_cols[i]] = abs(y_pred - y_test)

# Bootstrapping Linear Regression 30 times
acc_lg_list = []
num_iteration = 30
for i in range(num_iteration):       
    df_linear = pd.DataFrame(index= df_bsi_rpq_only.index, columns=rpq_cols + bsi_cols)
    run_lg( train_cols = rpq_cols, test_cols = bsi_cols)
    run_lg(train_cols = bsi_cols, test_cols = rpq_cols)
    acc_lg_list.append(df_linear.mean().mean())

#%%
df_DLSTS_orig = pd.DataFrame(index= df_bsi_rpq_only.index, columns=rpq_cols + bsi_cols)
df_DLSTS_mae_orig = pd.DataFrame(index= df_bsi_rpq_only.index, columns=rpq_cols + bsi_cols) 

def run_sts(model_input,train_cols,test_cols,inp_test,pred_test,score_file,df_acc):
            #,df_mae):
    X = df_bsi_rpq_only[train_cols]
    y = df_bsi_rpq_only[test_cols[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    if model_input == "STS":
        link_hists = False
        # Use df_STS
    elif model_input == "DL_STS":
        link_hists = True
        # Use df_DLSTS
    score_dict,text_dict,hist_dict,simil_arr = set_crosswalk_files(score_file=score_file,
                                                                    inv_in = inp_test,
                                                                   inv_out = pred_test)

    y_pred = np.ones(shape=(len(df_bsi_rpq_only),len(test_cols)))*-1
    for i in X_train.index:
    # for i in range(len(df_bsi_rpq_only)):
        predicted_scores = crosswalk_scores(df_bsi_rpq_only.loc[i,train_cols].values.astype(int), 
                                            score_dict, text_dict, hist_dict, simil_arr,
                                            inv_in = inp_test,inv_out = pred_test,verbose = False,
                                            empirical_shift_down = False,
                                            link_hists = link_hists, random_seed= i)
        y_pred[i,:] = np.asarray(predicted_scores)
        # y_pred[i,:] = 1 *(y_pred[i,:] > 3)
        y_test = df_bsi_rpq_only.loc[i,test_cols].values
        # y_test = 1 * (y_test > 3)
        df_acc.loc[i,test_cols] = (y_pred[i,:] == y_test).astype(int)
        # df_mae.loc[i,test_cols] = abs(y_pred[i,:] - y_test)


acc_sts_list = []
num_iteration = 30
for i in range(num_iteration):
    df_DLSTS_orig = pd.DataFrame(index= df_bsi_rpq_only.index, columns=rpq_cols + bsi_cols)
    run_sts(model_input = "DL_STS",train_cols=rpq_cols, test_cols=bsi_cols,
            inp_test="RPQ",pred_test="BSI",score_file='score_dict.p' ,df_acc=df_DLSTS_orig)
    run_sts(model_input = "DL_STS",train_cols=bsi_cols, test_cols=rpq_cols,
        inp_test="BSI",pred_test="RPQ",score_file='score_dict.p',df_acc=df_DLSTS_orig)
    acc_sts_list.append(df_DLSTS_orig.mean().mean())

#%%

# Plotting figure

fig, ax = plt.subplots()
acc_lists = [acc_lg_list, acc_sts_list]
bplot = ax.boxplot(acc_lists,
                   patch_artist=True,)  # fill with color
# fill with colors
for patch, color in zip(bplot['boxes'], ['orange','tomato']):
    patch.set_facecolor(color)

plt.show()