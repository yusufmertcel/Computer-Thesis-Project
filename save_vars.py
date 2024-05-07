# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:12:25 2023

@author: sceli
"""

import pickle
import pandas as pd

diff_day_params = {
    'RF': [
       {'n_estimators': 500,
       'min_samples_split': 11,
       'min_samples_leaf': 19,
       'max_features': 1.0,
       'max_depth': 19,
       'bootstrap': True,
       'random_state': 0}, 
      {'n_estimators': 500,
      'min_samples_split': 11,
      'min_samples_leaf': 19,
      'max_features': 1.0,
      'max_depth': 19,
      'bootstrap': True,
      'random_state': 0},
      {'n_estimators': 500,
      'min_samples_split': 11,
      'min_samples_leaf': 19,
      'max_features': 1.0,
      'max_depth': 19,
      'bootstrap': True,
      'random_state': 0}
      ],
    'GB': [ 
        {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth':26,
        'min_samples_split':5,
        'min_samples_leaf':12,
        'random_state':0
        },
        {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth':26,
        'min_samples_split':5,
        'min_samples_leaf':12,
        'random_state':0
        },
        {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth':26,
        'min_samples_split':5,
        'min_samples_leaf':12,
        'random_state':0
        }
        ],
    'XGB': [
        {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth':2,
        'min_child_weight':8,
        'subsample':0.5,
        'colsample_bytree':0.7,
        'random_state':0
        },
        {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth':2,
        'min_child_weight':8,
        'subsample':0.5,
        'colsample_bytree':0.7,
        'random_state':0
        },
        {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth':2,
        'min_child_weight':8,
        'subsample':0.5,
        'colsample_bytree':0.7,
        'random_state':0
        }
        ],
    'LGBM': [
       {
       'objective': 'regression',
       'boosting_type': 'gbdt',
       'n_estimators': 100,
       'learning_rate': 0.01,
       'max_depth': 6,
       'min_child_samples': 5,
       'subsample': 1.0,
       'colsample_bytree': 0.6,
       'reg_alpha': 0.01,
       'reg_lambda': 0,
       'n_jobs': -1,
       'random_state': 0
       },
        {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_samples': 5,
        'subsample': 1.0,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.01,
        'reg_lambda': 0,
        'n_jobs': -1,
        'random_state': 0
        },
        {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_samples': 5,
        'subsample': 1.0,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.01,
        'reg_lambda': 0,
        'n_jobs': -1,
        'random_state': 0
        }
        ],
    'CB': [
        {
        'iterations':500,
        'depth':4,
        'learning_rate':0.01,
        'l2_leaf_reg':3,
        'border_count':80,
        'bagging_temperature':0.5,
        'verbose':False
        },
        {
        'iterations':500,
        'depth':4,
        'learning_rate':0.01,
        'l2_leaf_reg':3,
        'border_count':80,
        'bagging_temperature':0.5,
        'verbose':False
        },
        {
        'iterations':500,
        'depth':4,
        'learning_rate':0.01,
        'l2_leaf_reg':3,
        'border_count':80,
        'bagging_temperature':0.5,
        'verbose':False
        }
        ]
    }


# # 29 Ekim
# diff_day_sets = [
#     (pd.date_range(start="10-22-2019", end="10-22-2019 23:55", freq="5T"),
#       ["04-22-2019","04-23-2019","10-23-2019","10-24-2019","10-26-2019","08-29-2019"],
#       [],
#       [],
#       pd.date_range(start="10-28-2019", end="10-29-2019 00:00", freq="5T")), # Tatil Öncesi
#     (pd.date_range(start="08-30-2018", end="08-30-2018 23:55", freq="5T"),
#      ["04-23-2019","10-22-2018","10-23-2018","10-24-2018","10-25-2018","10-26-2018","10-27-2018","10-28-2018","10-29-2018","10-30-2018",
# "10-31-2018","11-01-2018","11-02-2018","11-03-2018","11-04-2018",
# "10-21-2019","10-22-2019","10-23-2019","10-24-2019","10-25-2019","10-26-2019","10-27-2019","10-28-2019"],
#       ["05-01-2018","04-23-2018","05-19-2018","07-15-2018","08-30-2018","10-29-2018","01-01-2018"],
#       ["10-29-2019"],
#       pd.date_range(start="10-29-2019", end="10-30-2019 00:00", freq="5T")), # Tatil
#     (pd.date_range(start="10-09-2019", end="10-09-2019 23:55", freq="5T"), # 04-24-2019
#       ["10-16-2019","10-23-2019","10-10-2019","10-14-2019","10-07-2019","10-24-2019","10-16-2019"],
#       [],
#       [],
#       pd.date_range(start="10-30-2019", end="10-31-2019 00:00", freq="5T")) # Tatil Sonrası
#     ]

# # 23 Nisan
# diff_day_sets = [
#     (pd.date_range(start="04-12-2019", end="04-12-2019 23:55", freq="5T"),
#       ["04-08-2019","04-22-2018"], # 04-22-2018
#       [],
#       [],
#       pd.date_range(start="04-22-2019", end="04-23-2019 00:00", freq="5T")), # Tatil Öncesi
#     (pd.date_range(start="04-23-2018 00:00", end="04-23-2018 23:55", freq="5T"),
#       ["04-21-2019","05-01-2018","08-30-2018","10-28-2018","10-29-2018","04-28-2018"],
#       [],
#       ["04-23-2019"],
#       pd.date_range(start="04-23-2019", end="04-24-2019 00:00", freq="5T")), # Tatil
#     (pd.date_range(start="04-03-2019", end="04-03-2019 23:55", freq="5T"), # 04-24-2019
#       ["04-10-2019","04-17-2019"],
#       [],
#       [],
#       pd.date_range(start="04-24-2019", end="04-25-2019 00:00", freq="5T")) # Tatil Sonrası
#     ]

# 19 Mayıs
# diff_day_sets = [
#     (pd.date_range(start="04-27-2019", end="04-27-2019 23:55", freq="5T"),
#       ["05-04-2019","05-11-2019"], # "05-05-2018","04-28-2018" en kötü 05-04-2019 
#       [],
#       [],
#       pd.date_range(start="05-18-2019", end="05-19-2019 00:00", freq="5T")), # Tatil Öncesi
#     (pd.date_range(start="05-12-2019 00:00", end="05-12-2019 23:55", freq="5T"),
#       ["05-20-2018","05-12-2019","05-27-2018","05-18-2019"], # 
#       [],
#       [],
#       pd.date_range(start="05-19-2019", end="05-20-2019 00:00", freq="5T")), # Tatil
#     (pd.date_range(start="05-13-2019", end="05-13-2019 23:55", freq="5T"), # 04-24-2019
#       ["05-14-2019","05-15-2019","05-16-2019","05-17-2019","05-28-2018"], # "05-28-2018","05-21-2018","05-06-2019
#       [],
#       [],
#       pd.date_range(start="05-20-2019", end="05-21-2019 00:00", freq="5T")) # Tatil Sonrası
#     ]


# # 15 Temmuz

# diff_day_sets = [
#     (pd.date_range(start="06-23-2019", end="06-23-2019 23:55", freq="5T"),
#       ["06-30-2019","07-07-2019"], # 04-22-2018
#       [],
#       [],
#       pd.date_range(start="07-14-2019", end="07-15-2019 00:00", freq="5T")), # Tatil Öncesi
#     (pd.date_range(start="01-01-2019 00:00", end="01-01-2019 23:55", freq="5T"),
#       ["05-01-2019","07-14-2019","04-23-2019","07-13-2019","07-08-2019","04-30-2018","05-01-2018","04-22-2018","04-23-2018","05-18-2018","05-19-2018","07-14-2018","07-15-2018","08-29-2018","08-30-2018","10-28-2018","10-29-2018"],
#       ["04-23-2019","05-01-2019","05-01-2018","04-23-2018","05-19-2018","08-30-2018","10-29-2018","01-01-2018"],
#       ["07-15-2019"],
#       pd.date_range(start="07-15-2019", end="07-16-2019 00:00", freq="5T")), # Tatil
#     (pd.date_range(start="07-08-2019", end="07-08-2019 23:55", freq="5T"), # 04-24-2019
#       ["07-09-2019","07-10-2019","07-11-2019","07-12-2019"],
#       [],
#       [],
#       pd.date_range(start="07-16-2019", end="07-17-2019 00:00", freq="5T")) # Tatil Sonrası
#     ]

# KURBAN
# diff_day_sets = [
#     (pd.date_range(start="08-16-2018", end="08-16-2018 23:55", freq="5T"),
#       ["05-30-2019", "05-31-2019", "07-26-2019", "06-13-2018"],
#       [],
#       [],
#       pd.date_range(start="08-08-2019", end="08-10-2019 00:00", freq="5T")), # Tatil Öncesi
#     (pd.date_range(start="08-18-2018", end="08-26-2018 23:55", freq="5T"),
#       ["08-03-2019","08-04-2019"],
#       ["08-21-2018","08-22-2018","08-23-2018","08-24-2018"],
#       ["08-11-2019","08-12-2019","08-13-2019","08-14-2019"],
#       pd.date_range(start="08-10-2019", end="08-19-2019 00:00", freq="5T")), # Tatil
#     (pd.date_range(start="07-22-2019", end="07-22-2019 23:55", freq="5T"), # 04-24-2019
#       ["08-16-2018","08-27-2018", "08-28-2018"],
#       [],
#       [],
#       pd.date_range(start="08-19-2019", end="08-21-2019 00:00", freq="5T")) # Tatil Sonrası
#     ]

# RAMAZAN
diff_day_sets = [
    (pd.date_range(start="05-28-2019", end="05-28-2019 23:55", freq="5T"),
      ["04-22-2019","04-27-2018","06-13-2018"],
      [],
      [],
      pd.date_range(start="05-30-2019", end="06-01-2019 00:00", freq="5T")), # Tatil Öncesi
    (pd.date_range(start="06-14-2018", end="06-17-2018 23:55", freq="5T"),
      ["08-20-2018","08-21-2018","08-22-2018","08-23-2018","08-24-2018","08-25-2018","08-26-2018"],
      ["06-14-2018","06-15-2018","06-16-2018","06-17-2018"],
      ["06-03-2019","06-04-2019","06-05-2019","06-06-2019"],
      pd.date_range(start="06-01-2019", end="06-10-2019 00:00", freq="5T")), # Tatil
    (pd.date_range(start="05-28-2019", end="05-28-2019 23:55", freq="5T"), # 04-24-2019
      ["05-27-2019", "04-22-2019"],
      [],
      [],
      pd.date_range(start="06-10-2019", end="06-12-2019 00:00", freq="5T")) # Tatil Sonrası
    ]

with open('Ramazan_parametre.pkl', 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(diff_day_params, file) 
    
with open('Ramazan_days.pkl', 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(diff_day_sets, file) 