# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:36:00 2023

@author: sceli
"""
from model_train import get_pred, plot_model, best_and_worst
from preprocess import read_table, train_test_split
from analyse import get_feature_importance
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from statistics import mean
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingRegressor
from keras.layers import LSTM
import os
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam 


with open('Kurban_parametre.pkl', 'rb') as file: 
      
    # A new file will be created 
    diff_day_params = pickle.load(file) 
    
# with open('29Ekim_days.pkl', 'rb') as file: 
      
#      # A new file will be created 
#      diff_day_sets = pickle.load(file) 
    
path_dir = "Grafikler_30Aug_HA"
path = "D:/CE 3.sınıf/Spring Semester/Ara Proje/Data/Documents/Speed Data"
os.chdir(path + "/2018")
l_2018 = os.listdir()
os.chdir(path + "/2019")
l_2019 = os.listdir()

results = pd.DataFrame({}, index= ["Historical Average"])
os.chdir("D:\CE 3.sınıf\Spring Semester\Ara Proje\Results")
if not os.path.exists(path_dir):
    os.mkdir(path_dir)

# 106_segments_v1.csv
segments = pd.read_csv(r"D:\CE 3.sınıf\Spring Semester\Ara Proje\Data\106_segments_v1.csv", names=['segment'])
#segments = segments.loc[[t.startswith("128_0") for t in segments["segment"].values]]
segments = segments.loc[segments["segment"] != '472_0']
l_2018 = [k for k in l_2018 if k.split('_2018')[0] in segments.values]
l_2019 = [k for k in l_2019 if k.split('_2019')[0] in segments.values]

# KURBAN GENEL PARAMETRE
diff_day_sets = [
    (pd.date_range(start="08-22-2019", end="08-28-2019 23:55", freq="5T"),
      [],
      [],
      [],
      pd.date_range(start="08-29-2019", end="08-31-2019 23:55", freq="5T")), # Tatil
    ]


param_rf = diff_day_params['RF']
for file_2018, file_2019 in zip(l_2018, l_2019):
    combined_pred_df = list()
    for index, day_set in enumerate(diff_day_sets):
        stacked_preds = list()
        error_rates = list()
        path_2018 = path+"/2018/"+file_2018
        path_2019 = path+"/2019/"+file_2019
        
        aralik, days, tatil, tatil_test, aralik_test = day_set
        
        
        
        # # Veri ön işleme adımlarını yap
        df, df_tahmin, df_DNN = read_table(path_2018, path_2019, aralik, days, tatil, tatil_test, aralik_test)
        #X_train, y_train, y_train_high, X_test, y_test = train_test_split(df, df_tahmin)
        df.set_index("tarih", inplace=True)
        df.dropna(inplace=True)
        df_tahmin.dropna(inplace=True)
        # Historical Average Kendi Modelimiz
        history = np.reshape(df["hiz"].values[:int(len(df)/288)*288], (int(len(df)/288), 288))
        test = np.reshape(df_tahmin["hiz"].values[:int(len(df_tahmin)/288)*288], (int(len(df_tahmin)/288), 288))
        predictions = list()
        for t in range(len(test)):
             #model = ARIMA(history, order=(5,1,0))
             #model_fit = model.fit()
             #output = model_fit.forecast()
             #yhat = output[0]
             yhat = np.mean(history, axis=0)
             history = np.vstack([history[1:], test[t]])
             
             predictions.append(yhat)
             #print('predicted=%f, expected=%f' % (yhat, obs))

        stacked_preds.append(np.asarray(predictions).flatten())
        # # 0 - SVR
        # # y_train - y_train_high
        # regr = make_pipeline(SVR(C=10000, gamma=0.1,epsilon=0.3))
        # y_pred_residual, mape = get_pred(regr, X_train, y_train, X_test, y_test, None)
        # print(f"SVR: {mape}")
        # #df_tahmin["hiz_tahmin"] = y_pred_residual
        # #df_tahmin = df_tahmin[["dayofyear", "saat", "hiz", "hiz_tahmin"]]
        # #plot_model("SVR", df_tahmin, path_dir, file_2018.split('_2018')[0])
        # stacked_preds.append(y_pred_residual)
        
        
        # feature_names = list(X_train.columns.values)
        # y_pred_residual = None
        # params = param_rf[1]
        # model = RandomForestRegressor(**params)
        # y_pred, mape = get_pred(model, X_train, y_train, X_test, y_test, y_pred_residual)
        # print(best_and_worst(y_test,y_pred).sort_values('abs_error', ascending=True).head(10))
        
        # stacked_preds.append(y_pred)
        
        #get_feature_importance(model, feature_names, X_test, y_test,  path_dir, file_2018.split('_2018')[0])
        
        
        a = pd.concat([pd.Series(x) for x in stacked_preds], axis=1)
        a["Speed"] = df_tahmin["hiz"].values[:int(len(df_tahmin)/288)*288]
        a.columns = ["Historical Average","Speed"]
        a.index = df_tahmin.index[:int(len(df_tahmin)/288)*288]
        
        combined_pred_df.append(a)
        
    total_pred_res = pd.concat(combined_pred_df, axis=0)
    total_pred_res.sort_index(inplace=True)
    for col in a.drop('Speed', axis=1).columns:
        mape = mean_absolute_percentage_error(total_pred_res['Speed'].values, total_pred_res[col].values)
        print(f"{col} - Hata oranı: {mape:.2f}")
        error_rates.append(mape)
        plot_model(col, total_pred_res[[col, 'Speed']], path_dir, file_2018.split('_2018')[0])
        
    results[file_2018] = error_rates
results["Average"] = results.mean(axis=1)
results.to_excel("Results_30Aug_HA.xlsx",index=True)