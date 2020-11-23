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

app_train = pd.read_csv("C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/application_train.csv").replace({365243:np.nan})
app_test = pd.read_csv("C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/application_test.csv").replace({365243:np.nan})
bureau = pd.read_csv('C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/bureau_balance.csv').replace({365243: np.nan})
cash = pd.read_csv('C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/POS_CASH_balance.csv').replace({365243: np.nan})
credit = pd.read_csv('C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/credit_card_balance.csv').replace({365243: np.nan})
previous = pd.read_csv('C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/previous_application.csv').replace({365243: np.nan})
installments = pd.read_csv('C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/predict-loan-repayment/installments_payments.csv').replace({365243: np.nan})

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




# Data manipulation
import pandas as pd
import numpy as np

# modeling
import lightgbm as lgb

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluating dictionary
import ast

RSEED = 50


features = pd.read_csv("C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/feature_matrix_article.csv")

#features = pd.read_csv("C:/Users/ATazo/Desktop/Loan_Repayment_Feature_Engineering/feature_matrix.csv")



def format_data(features):
    """Format a set of training and testing features joined together
       into separate sets for machine learning"""
    
    train = features[features['TARGET'].notnull()].copy()
    test = features[features['TARGET'].isnull()].copy()
    
    train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))
    test_ids = list(test['SK_ID_CURR'])
    
    train = train.drop(columns = ['TARGET', 'SK_ID_CURR'])
    test = test.drop(columns = ['TARGET', 'SK_ID_CURR'])
    
    feature_names = list(train.columns)
    
    return train, train_labels, test, test_ids, feature_names



def evaluate(fm, hyp_results):
    """Evaluate a feature matrix using the hyperparameter tuning results.
    
    Parameters:
        fm (dataframe): feature matrix with observations in the rows and features in the columns. This will
                        be passed to `format_data` and hence must have a train set where the `TARGET` values are 
                        not null and a test set where `TARGET` is null. Must also have the `SK_ID_CURR` column.
        
        hyp_results (dataframe): results from hyperparameter tuning. Must have column `score` (where higher is better)
                                 and `params` holding the model hyperparameters
                                 
    Returns:
        results (dataframe): the cross validation roc auc from the default hyperparameters and the 
                             optimal hyperparameters
        
        feature_importances (dataframe): feature importances from the gradient boosting machine. Columns are 
                                          `feature` and `importance`. This can be used in `plot_feature_importances`.
                                          
        submission (dataframe): Predictions which can be submitted to the Kaggle Home Credit competition. Save
                                these as `submission.to_csv("filename.csv", index = False)` and upload
       """
    
    print('Number of features: ', (fm.shape[1] - 2))

    # Format the feature matrix 
    train, train_labels, test, test_ids, feature_names = format_data(fm)
    
    # Training set 
    train_set = lgb.Dataset(train, label = train_labels)


    # Dataframe to hold results
    results = pd.DataFrame(columns = ['default_auc', 'default_auc_std', 
                                      'opt_auc', 'opt_auc_std', 
                                      'random_search_auc'], index = [0])

    # Create a default model and find the hyperparameters
    model = lgb.LGBMClassifier()
    default_hyp = model.get_params()
    


    # Remove n_estimators because this is found through early stopping
    del default_hyp['n_estimators'], default_hyp['silent']


    # Cross validation with default hyperparameters
    default_cv_results = lgb.cv(default_hyp, train_set, nfold = 5, num_boost_round = 10000, early_stopping_rounds = 100, 
                                metrics = 'auc', seed = RSEED)

from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
lbl.fit(train["NAME_TYPE_SUITE"])

np.dtype(train["NAME_TYPE_SUITE"])
sample = lbl.transform(train_set)

train = train.drop(columns = ["NAME_TYPE_SUITE", "OCCUPATION_TYPE", "HOUSETYPE_MODE", "ORGANIZATION_TYPE", "FLAG_OWN_REALTY",
                           "NAME_INCOME_TYPE", "FLAG_OWN_CAR", "FONDKAPREMONT_MODE", "NAME_CONTRACT_TYPE", "NAME_FAMILY_STATUS", 
                           "CODE_GENDER", "WEEKDAY_APPR_PROCESS_START", "NAME_HOUSING_TYPE", "EMERGENCYSTATE_MODE", "WALLSMATERIAL_MODE",
                           "NAME_EDUCATION_TYPE", "MODE(previous_app.NAME_CONTRACT_TYPE)", "MODE(bureau_balance.STATUS)",
                           "MODE(bureau.CREDIT_CURRENCY)", "MODE(previous_app.NAME_GOODS_CATEGORY)", "MODE(bureau.CREDIT_ACTIVE)", 
                           "MODE(previous_app.FLAG_LAST_APPL_PER_CONTRACT)", "MODE(previous_app.NAME_CLIENT_TYPE)", 
                           "MODE(previous_app.CODE_REJECT_REASON)", "MODE(previous_app.WEEKDAY_APPR_PROCESS_START)", 
                           "MODE(previous_app.NAME_PAYMENT_TYPE)", "MODE(previous_app.NAME_PRODUCT_TYPE)", 
                           "MODE(cash.NAME_CONTRACT_STATUS)", "MODE(previous_app.PRODUCT_COMBINATION)", 
                           "MODE(previous_app.NAME_PORTFOLIO)", "MODE(previous_app.CHANNEL_TYPE)", 
                           "MODE(previous_app.NAME_CASH_LOAN_PURPOSE)", "MODE(previous_app.NAME_CONTRACT_STATUS)", 
                           "MODE(previous_app.NAME_TYPE_SUITE)", "MODE(previous_app.NAME_YIELD_GROUP)", "MODE(credit.NAME_CONTRACT_STATUS)",
                           "MODE(bureau.CREDIT_TYPE)", "MODE(previous_app.NAME_SELLER_INDUSTRY)",
                           "MODE(previous_app.MODE(credit.NAME_CONTRACT_STATUS))", "MODE(previous_app.MODE(cash.NAME_CONTRACT_STATUS))", 
                           "MODE(bureau.MODE(bureau_balance.STATUS))"]) 
    
train = train.drop(columns = ["CODE_GENDER", "EMERGENCYSTATE_MODE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "NAME_CONTRACT_TYPE", 
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_TYPE_SUITE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "WALLSMATERIAL_MODE", "WEEKDAY_APPR_PROCESS_START", "MODE(previous.NAME_CONTRACT_TYPE)", "MODE(previous.WEEKDAY_APPR_PROCESS_START)", 
    "MODE(previous.FLAG_LAST_APPL_PER_CONTRACT)", "MODE(previous.NAME_CASH_LOAN_PURPOSE)", "MODE(previous.NAME_CONTRACT_STATUS)", 
    "MODE(previous.NAME_PAYMENT_TYPE)", "MODE(previous.CODE_REJECT_REASON)", "MODE(previous.NAME_TYPE_SUITE)", 
    "MODE(previous.NAME_CLIENT_TYPE)", "MODE(previous.NAME_GOODS_CATEGORY)", "MODE(previous.NAME_PORTFOLIO)", 
    "MODE(previous.NAME_PRODUCT_TYPE)","MODE(previous.CHANNEL_TYPE)", "MODE(previous.NAME_SELLER_INDUSTRY)", 
    "MODE(previous.NAME_YIELD_GROUP)", "MODE(previous.PRODUCT_COMBINATION)", "MODE(bureau.CREDIT_ACTIVE)", 
    "MODE(bureau.CREDIT_CURRENCY)", "MODE(bureau.CREDIT_TYPE)", "MODE(cash.NAME_CONTRACT_STATUS)", 
    "MODE(credit.NAME_CONTRACT_STATUS)", "MODE(previous.MODE(cash.NAME_CONTRACT_STATUS))", 
    "MODE(previous.MODE(credit.NAME_CONTRACT_STATUS))", "MODE(bureau.MODE(bureau_balance.STATUS))"])
    
MODE(bureau_balance.STATUS)'] 
    
    default_auc = default_cv_results['auc-mean'][-1]
    default_auc_std = default_cv_results['auc-stdv'][-1]
    
    # Locate the optimal hyperparameters
    hyp_results = hyp_results.sort_values('score', ascending = False).reset_index(drop = True)
    best_hyp = ast.literal_eval(hyp_results.loc[0, 'params'])
    best_random_score = hyp_results.loc[0, 'score']

    del best_hyp['n_estimators']

    # Cross validation with best hyperparameter values
    opt_cv_results = lgb.cv(best_hyp, train_set, nfold = 5, num_boost_round = 10000, early_stopping_rounds = 100, 
                            metrics = 'auc', seed = RSEED)

    opt_auc = opt_cv_results['auc-mean'][-1]
    opt_auc_std = opt_cv_results['auc-stdv'][-1]
    
    # Insert results into dataframe
    results.loc[0, 'default_auc'] = default_auc
    results.loc[0, 'default_auc_std'] = default_auc_std
    results.loc[0, 'random_search_auc'] = best_random_score
    results.loc[0, 'opt_auc'] = opt_auc
    results.loc[0, 'opt_auc_std'] = opt_auc_std
    
    # Extract the optimum number of estimators
    opt_n_estimators = len(opt_cv_results['auc-mean'])
    model = lgb.LGBMClassifier(n_estimators = opt_n_estimators, **best_hyp)
    
    # Fit on whole training set
    model.fit(train, train_labels)

    # Make predictions on testing data
    preds = model.predict_proba(test)[:, 1]

    # Make submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 
                               'TARGET': preds})

    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype(np.int32)
    
    # Make feature importances dataframe
    feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': model.feature_importances_})

    return results, feature_importances, submission