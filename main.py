from model_train import get_pred, plot_model, best_and_worst
from preprocess import read_table, train_test_split
from analyse import get_feature_importance
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
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
import streamlit as st


excel_tablo_ismi = "Results_Ramazan_13Gun_diff_daysets_TumTatil_SON_KESIN_v2.xlsx"
grafik_klasor_ismi = "Grafikler_Ramazan_13Gun_SonRapor_KESIN_v2"

# r"D:\CE 3.sınıf\Spring Semester\Ara Proje\Data\106_segments_v1.csv"
segments = pd.read_csv("106_segments_v1.csv", names=['segment'])

st.title("Short and Long Holidays Traffic Speed Prediction")
    
# Description
st.text("Choose a Holiday and a Segment in the sidebar. Input your values and get a prediction.")
days = ["1Mayis", "15Tem", "19Mayis", "23Nisan", "29Ekim", "30Aug", "Kurban", "Ramazan"]
#sidebar
sideBar = st.sidebar
day = sideBar.selectbox('Which Holiday do you want to predict?',days)
segment_Names = ["All Segments"] + segments["segment"].values.tolist()
segment_Name = sideBar.selectbox('Which segment do you want to use?',segment_Names)

result = st.button("Run Model")
stop = st.button("Stop Model")
if result:
    with open(f'{day}_parametre.pkl', 'rb') as file: 
          
        # A new file will be created 
        diff_day_params = pickle.load(file) 
        
        
    with open(f'{day}_days.pkl', 'rb') as file: 
          
        # A new file will be created 
        diff_day_sets = pickle.load(file) 
    
    # Veri dosya yolu ve klasörleri tanımla
    path = os.getcwd()
    #path = "D:/CE 3.sınıf/Spring Semester/Ara Proje/Data/Documents/Speed Data"
    os.chdir(path + "/2018")
    l_2018 = os.listdir()
    os.chdir(path + "/2019")
    l_2019 = os.listdir()
    
    path_dir = grafik_klasor_ismi
    
    # Veri dosya yolu ve klasörleri tanımla
    results = pd.DataFrame({}, index= ["RandomForestRegressor","XGBRegressor", "CatBoost","Ensemble"])
    
    #os.chdir("D:\CE 3.sınıf\Spring Semester\Ara Proje\Results")
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    # 106_segments_v1.csv
    if segment_Name != "All Segments":
        segments = segments.loc[[t.startswith(segment_Name) for t in segments["segment"].values]]
    l_2018 = [k for k in l_2018 if k.split('_2018')[0] in segments.values]
    l_2019 = [k for k in l_2019 if k.split('_2019')[0] in segments.values]
    
       
    
    
    
    param_rf, param_gb, param_xgb, param_lgbm, param_cb = diff_day_params['RF'], diff_day_params['GB'], diff_day_params['XGB'], diff_day_params['LGBM'], diff_day_params['CB']
    # Her bir veri dosyası için döngü
    all_segments_speed = list()
    for file_2018, file_2019 in zip(l_2018, l_2019):
        combined_pred_df = list()
        for index, day_set in enumerate(diff_day_sets):
            stacked_preds = list()
            error_rates = list()
            path_2018 = path+"/2018/"+file_2018
            path_2019 = path+"/2019/"+file_2019
            
            aralik, days, tatil, tatil_test, aralik_test = day_set
            # Veri ön işleme adımlarını yap
            df, df_tahmin, df_DNN = read_table(path_2018, path_2019, aralik, days, tatil, tatil_test, aralik_test)
            X_train, y_train, y_train_high, X_test, y_test = train_test_split(df, df_tahmin)
            X_train_DNN, y_train_DNN, _, X_test_DNN, y_test_DNN = train_test_split(df_DNN, df_tahmin)
            
            feature_names = list(X_train.columns.values)
         #----------------------------------------------------------------------------------------- 
         # Farklı algoritmaları kullanarak tahminler yap
            
            params = param_rf[index]
            # 1 -RandomForestRegressor 
            # Model oluştur ve eğit
            model = RandomForestRegressor(**params)
            y_pred, mape = get_pred(model, X_train, y_train, X_test, y_test)
            print(best_and_worst(y_test,y_pred).sort_values('abs_error', ascending=True).head(10))
            stacked_preds.append(y_pred)
    
            get_feature_importance(model, feature_names, X_test, y_test,  path_dir, file_2018.split('_2018')[0])
            
            # 3 -XGBRegressor 
            params = param_xgb[index]
            # Model oluşturma ve eğitme
            model = XGBRegressor(**params)
            y_pred, mape = get_pred(model,X_train, y_train, X_test, y_test)
           
            stacked_preds.append(y_pred)
            print(best_and_worst(y_test,y_pred).sort_values('abs_error', ascending=True).head(10))
            
            # 5-CatBoost 
            params = param_cb[index]
            # Model oluştur ve eğit
            model = CatBoostRegressor(**params)
    
            y_pred, mape = get_pred(model,X_train, y_train, X_test, y_test)
            stacked_preds.append(y_pred)
            print(best_and_worst(y_test,y_pred).sort_values('abs_error', ascending=True).head(10))
            
            # Ensemble
            stacked_preds_avg = np.average(stacked_preds, axis=0)
            stacked_preds.append(stacked_preds_avg) # bütün modellerin tahminlerinin list of listi
            
            a = pd.concat([pd.Series(x) for x in stacked_preds], axis=1)
            a["Speed"] = y_test.values
            a.columns = ["RandomForestRegressor", "XGBRegressor","CatBoost","Ensemble","Speed"]
            a.index = df_tahmin.index
            
            combined_pred_df.append(a)
            
        total_pred_res = pd.concat(combined_pred_df, axis=0)
        total_pred_res.sort_index(inplace=True)
        for col in a.drop('Speed', axis=1).columns:
            mape = mean_absolute_percentage_error(total_pred_res['Speed'].values, total_pred_res[col].values)
            print(f"{col} - Hata oranı: {mape:.2f}")
            error_rates.append(mape)
            figure = plot_model(col, total_pred_res[[col, 'Speed']], path_dir, file_2018.split('_2018')[0])
            st.pyplot(figure)
            st.write(f"{col} - Hata oranı: {mape:.2f}")
        all_segments_speed.append(total_pred_res[["Ensemble","Speed"]])
        results[file_2018] = error_rates
    
    all_segments_as_AVG = all_segments_speed[0]
    for i in range(1,len(segments)):
        all_segments_as_AVG = (all_segments_as_AVG[["Ensemble","Speed"]] + all_segments_speed[i][["Ensemble","Speed"]])
    
    all_segments_as_AVG /= len(all_segments_speed)
    all_segments_as_AVG.index = all_segments_speed[0].index 
    figure = plot_model("Average_Speed_All_Segments", all_segments_as_AVG, path_dir, "_106")
    st.pyplot(figure)
    all_segments_as_AVG.dropna(inplace=True)
    mape = mean_absolute_percentage_error(all_segments_as_AVG['hiz'].values, all_segments_as_AVG['hiz_tahmin'].values)
    print(f"All Segment MAPE: {mape}")
    st.write(f"All Segment MAPE: {mape}")
    results["Average"] = results.mean(axis=1)
    results.to_excel(excel_tablo_ismi,index=True)

elif stop:
    st.write("Model is stopped")
    os.chdir("D:/CE 3.sınıf/Spring Semester/Ara Proje/model")
    
    
