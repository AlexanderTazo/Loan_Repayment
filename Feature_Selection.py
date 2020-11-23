# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:54:59 2019

@author: ATazo

Feature-Selection and Optimal Parameters
"""


import pandas as pd
import numpy as np

#Featuretools-features

feature_matrix = pd.read_csv("../predict-loan-repayment/feature_matrix.csv")

#Sample 10% of the data or else it takes too long time to process
train = feature_matrix[feature_matrix["TARGET"].notnull()].sample(frac = 0.1, random_state=50)

import gc
gc.enable()
del feature_matrix
gc.collect()


#Correct the column types

for col in ["SUM(bureau.PREVIOUS_OTHER_LOAN_RATE)", "SUM(bureau.PREVIOUS_OTHER_LOAN_RATE WHERE CREDIT ACTIVE = Closed)",
            "SUM(bureau.PREVIOUS_OTHER_LOAN_RATE WHERE CREDIT ACTIVE = Active)", "SUM(bureau_balance.bureau.PREVIOUS_OTHER_LOAN_RATE)"]:
    try:
        train[col] = train[col].astype(np.float32)
    except:
        print(f"{col} not in data")
        
        
for col in train:
    if train[col].dtype == "bool":
        train[col] = train[col].astype(np.uint8)


train = pd.get_dummies(train)
n_features_start = train.shape[1] - 2
train.shape

#Columns with duplicated values

x, idx, inv, counts = np.unique(train, axis = 1, return_index = True, return_inverse = True,
                                return_counts = True, )

train = train.iloc[:, idx]
n_non_unique_columns = n_features_start - train.shape[1] - 2
train.shape

#Missing values- remove columns with more than 90% missing

missing_threshold = 90

missing = pd.DataFrame(train.isnull().sum())
missing["percent"] = 100*(missing[0] / train.shape[0])
missing.sort_values("percent", ascending = False, inplace = True)

missing_cols = list(missing[missing["percent"] > missing_threshold].index )
n_missing_cols = len(missing_cols)

train = train[[x for x in train if x not in missing_cols]]


#Zero Variance Columns

unique_counts = pd.DataFrame(train.nunique()).sort_values(0,ascending= True)
zero_var_cols = list(unique_counts[unique_counts[0] == 1].index)
n_zero_var_cols = len(zero_var_cols)

train = train[[x for x in train if x not in zero_var_cols]]

#Remove any derivations of Target if by accident created

for col in train:
    if "TARGET" in col:
        print(col)

#Find Collinear Variables with Correlation Threshold

correlation_threshold = 0.95     

corr_matrix = train.corr()

# Extract upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

train = train[[x for x in train if x not in to_drop]]

#Save the sample data to CSV-file
train.to_csv("../predict-loan-repayment/feature_matrix_sample.csv", index = False)


#Lastly we want to make a function that applies the four steps above in same sequence

def feature_selection(feature_matrix, missing_threshold = 90, correlation_threshold = 0.95):
    """Feature Selection for a dataframe"""
    
    feature_matrix = pd.get_dummies(feature_matrix)
    n_features_start = feature_matrix.shape[1]
    print("Original shape", feature_matrix.shape)
    
    _, idx = np.unique(feature_matrix, axis = 1, return_index = True)
    feature_matrix = feature_matrix.iloc[:, idx]
    n_non_unique_columns = n_features_start - feature_matrix.shape[1]
    print("{} non-unique valued columns.".format(n_non_unique_columns))
    
    #Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing["percent"] = 100*(missing[0]/feature_matrix.shape[0])
    missing.sort_values("percent", ascending = True, inplace = True)
    
    #Missing above threshold
    missing_cols = list(missing[missing["percent"] > missing_threshold].index)
    n_missing_cols = len(missing_cols)
    
    #Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if not in missing_cols]]
    print("{} missing columns with threshold:{}.".format(n_missing_cols, missing_threshold))
    
    #Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)
    
    #Remove zero variance cols
    feature_matrix = feature_matrix[[x for x in feature_matrix if not in zero_variance_cols]]
    print("{} zero variance columns".format(n_zero_variance_cols))
    
    #Correlations
    corr_matrix = feature_matrix.corr()
    
    #Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k =1).astype(np.bool))
    
    #Select features with correlations above threshold
    #Need to use absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    
    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print("{} collinear columns removed with threshold {}".format(n_collinear, correlation_threshold))
    
    total_removed = n_non_unique_columns + n_missing_cols + +n_zero_variance_cols + n_collinear
    
    print("Total columns removed", total_removed)
    print("Shape after feature selection: {}.".format(feature_matrix.shape))
    
    return feature_matrix


# Automated Engineering Features
    
sample = pd.read_csv("../predict-loan-repayment/feature_matrix_sample.csv")
fm = pd.read_csv("../predict-loan-repayment/feature_matrix.csv")

cat = pd.get_dummies(fm.select_dtypes("object"))

for col in fm:
    if fm[col].dtype == "bool":
        fm[col] == fm[col].astype(np.uint8)
        
fm = fm.select_dtypes(["number"])
fm = pd.concat([fm,cat], axis = 1)

fm = fm[[x for x in sample.columns if x in fm]]

fm.to_csv("../predict-loan-repayment/feature_matrix_selected.csv", index = False)   






