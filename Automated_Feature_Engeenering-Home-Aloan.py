# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:14:06 2019

@author: ATazo

Prediction Loan Repayment Home Aloan
"""

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

app_train = pd.read_csv("../predict-loan-repayment/application_train.csv").replace({365243:np.nan})
app_test = pd.read_csv("..(predict-loan-repayment/application_test.csv").replace({365243:np.nan})
bureau = pd.read_csv("../predict-loan-repayment/bureau.csv").replace({365243: np.nan})
bureau_balance = pd.read_csv('../predict-loan-repayment/bureau_balance.csv").replace({365243: np.nan})
cash = pd.read_csv("../predict-loan-repayment/POS_CASH_balance.csv").replace({365243: np.nan})
credit = pd.read_csv("../predict-loan-repayment/credit_card_balance.csv").replace({365243: np.nan})
previous = pd.read_csv("../predict-loan-repayment/previous_application.csv").replace({365243: np.nan})
installments = pd.read_csv("../predict-loan-repayment/installments_payments.csv").replace({365243: np.nan})

app_test["TARGET"] = np.nan

app = app_train.append(app_test, ignore_index = True, sort = True)

for index in ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]:
    for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
        if index in list(dataset.columns):
            dataset[index]= dataset[index].fillna(0).astype(np.int64)
            

es = ft.EntitySet(id="clients")


import featuretools.variable_types as vtypes

app_types = {}

for col in app:
    if (app[col].nunique() == 2) and (app[col].dtype == float):
        app_types[col] = vtypes.Boolean

del app_types["TARGET"]

print ( "There are {} Boolean variables in the application data.".format(len(app_types)))

app_types["REGION_RATING_CLIENT"] = vtypes.Ordinal
app_types["REGION_RATING_CLIENT_W_CITY"] = vtypes.Ordinal
app_types["HOUR_APPR_PROCESS_START"] = vtypes.Ordinal

previous_types = {}

for col in previous:
    if(previous[col].nunique() == 2) and (previous[col].dtype == float):
        previous_types[col] = vtypes.Boolean
        
print('There are {} Boolean variables in the previous data.'.format(len(previous_types)))

installments = installments.drop(columns = ['SK_ID_CURR'])
credit = credit.drop(columns = ['SK_ID_CURR'])
cash = cash.drop(columns = ['SK_ID_CURR'])

es = es.entity_from_dataframe(entity_id= "app", dataframe=app, index = "SK_ID_CURR", 
                              variable_types= app_types)

es = es.entity_from_dataframe(entity_id= "bureau", dataframe = bureau, index = "SK_ID_BUREAU")

es = es.entity_from_dataframe(entity_id= "previous", dataframe=previous, index = "SK_ID_PREV",
                              variable_types= previous_types)

es = es.entity_from_dataframe(entity_id="bureau_balance", dataframe = bureau_balance, 
                              make_index= True, index = "bureaubalance_index")

es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, 
                              make_index = True, index = 'cash_index')

es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'installments_index')

es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index')

es


print("Parent: app, Parent Variable of bureau: SK_ID_CURR\n\n", app.iloc[:, 111:115].head())
print("\nChild: bureau, Child Variable of app: SK_ID_CURR\n\n", bureau.iloc[:,:5].head())

print('Parent: bureau, Parent Variable of bureau_balance: SK_ID_BUREAU\n\n', bureau.iloc[:, :5].head())
print('\nChild: bureau_balance, Child Variable of bureau: SK_ID_BUREAU\n\n', bureau_balance.head())


r_app_bureau = ft.Relationship(es["app"]["SK_ID_CURR"],es["bureau"]["SK_ID_CURR"])

r_bureau_balance = ft.Relationship(es["bureau"]["SK_ID_BUREAU"], es["bureau_balance"]["SK_ID_BUREAU"])

r_app_previous = ft.Relationship(es["app"]["SK_ID_CURR"], es["previous"]["SK_ID_CURR"])

r_r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

es = es.add_relationships([r_app_bureau,r_app_previous, r_bureau_balance,r_previous_credit,
                          r_previous_installments,r_r_previous_cash])

es

primitives = ft.list_primitives()

pd.options.display.max_colwidth = 100

default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]
feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives, 
                       where_primitives = [], seed_features = [],
                       max_depth = 2, n_jobs = -1, verbose = 1,
                       features_only=True)

ft.save_features(feature_names, '../input/features.txt')


#Run Deep-Feature Synthesis- will take a lot of time to process- use pararell processors

print('Total size of entityset: {:.5f} gb.'.format(sys.getsizeof(es) / 1e9))

import psutil

print('Total number of cpus detected: {}.'.format(psutil.cpu_count()))
print('Total size of system memory: {:.5f} gb.'.format(psutil.virtual_memory().total / 1e9))


# feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='app',
#                                        agg_primitives = agg_primitives,
#                                        trans_primitives = trans_primitives,
#                                        seed_features = seed_features,
#                                         where_features = where_features,
#                                        n_jobs = 1, verbose = 1, features_only = False,
#                                        max_depth = 2, chunk_size = 100)

# feature_matrix.reset_index(inplace = True)
# feature_matrix.to_csv('../input/feature_matrix.csv', index = False)



