# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:12:46 2019

@author: ATazo

"""
"""Random Search for Optimal Hyperparameters"""

import pandas as pd
import numpy as np

import lightgbm as lgb

from timeit import default_timer as timer

import csv

import random


fm = pd.read_csv("../predict-loan-repayment/feature_matrix.csv")  
fm = fm.sample(frac = 0.1, random_state=50)


train = features[features["set"] == "train"]
train = train[[x for x in sample if x in train]]

    
test = features[features["set"] == "test"]
test = test[[x for x in sample.columns if x in test]]

train_labels = np.array(train["TARGET"].astype(np.int32)).reshape((-1))
test_ids = list(test_1["SK_ID_CURR"])

feature_names = list(train.columns)

train = train.drop(columns = ['TARGET', 'SK_ID_CURR', "set"])
test = test.drop(columns = ['TARGET', 'SK_ID_CURR', "set"])


train_set = lgb.Dataset(train, label = train_labels)

results = pd.DataFrame(columns = ['default_auc', 'default_auc_std', 
                                      'opt_auc', 'opt_auc_std', 
                                      'random_search_auc'], index = [0])
model = lgb.LGBMClassifier()
default_hyp = model.get_params()

del default_hyp['n_estimators'], default_hyp['silent']

default_cv_results = lgb.cv(default_hyp, train_set, nfold = 5, num_boost_round = 10000, early_stopping_rounds = 100, 
                                metrics = 'auc', seed = 50)

print("Cross Validation ROC AUC: {:.5f} with std{:.5f}.".format(default_cv_results["auc-mean"][-1],
      default_cv_results["auc-stdv"][-1]))

print("Number of estimators trained: {}".format(len(default_cv_results["auc-mean"])))

model = lgb.LGBMClassifier(n_estimators = 46, random_state = 50)
model.fit(train, train_labels)

preds = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({"SK_ID_CURR": test_ids,
                          "TARGET": preds})


#Now that we have created a baseline accuracy with feature_matrix with x for x in sample lets optimize hyperparameters

#Function for the above steps to make it easier to do random search
N_FOLDS = 5
MAX_EVALS = 100

def objective(hyperparameters, iteration):
    """Objective function for random search. Returns
       the cross validation score from a set of hyperparameters."""

    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']

    start = timer()
     # Perform n_folds cross validation with early stopping
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000,
                         nfold = N_FOLDS,
                        early_stopping_rounds = 100, metrics = 'auc')

    time = timer() - start
    # Best score is last in cv results
    score = cv_results['auc-mean'][-1]
    std = cv_results['auc-stdv'][-1]

    # Number of estimators os length of results
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators

    return [score, std, hyperparameters, iteration, time]




def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter tuning"""

    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'std', 'params', 'iteration', 'time'],
                                  index = list(range(MAX_EVALS)))


    for i in range(MAX_EVALS):

        # Choose random hyperparameters
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        # Set correct subsample
        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(random_params, i)
        results.loc[i, :] = eval_results

       

    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)

    return results


   # Hyperparameter grid
    param_grid = {
        'is_unbalance': [True, False],
        'boosting_type': ['gbdt', 'goss', 'dart'],
        'num_leaves': list(range(20, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
        'subsample_for_bin': list(range(20000, 300000, 20000)),
        'min_child_samples': list(range(20, 500, 5)),
        'reg_alpha': list(np.linspace(0, 1)),
        'reg_lambda': list(np.linspace(0, 1)),
        'colsample_bytree': list(np.linspace(0.6, 1, 10)),
        'subsample': list(np.linspace(0.5, 1, 100))
    }    


        # Write results to line of file
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(eval_results)

        # Make sure to close file
        of_connection.close()
out_file = '../predict-loan-repayment/%s.csv'

headers = ["score", "std", "hyperparameters", "iteration", "time"]
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(headers)
    of_connection.close()

    
    