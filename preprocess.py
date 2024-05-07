import pandas as pd
import numpy as np
from datetime import time
import matplotlib.pyplot as plt
from dft import low_passFilter
from sklearn.cluster import KMeans


def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]


def read_table(file_1,file_2, aralik, days, tatil, tatil_test, aralik_test):
    
    # CSV dosyalarını oku
    df_2018 = pd.read_csv(file_1, names= ["tarih", "hiz"], header=0)
    df_2019 = pd.read_csv(file_2, names= ["tarih", "hiz"], header=0)
    
    # Tarih sütunlarını datetime formatına çevir
    df_2018['tarih'] = pd.to_datetime(df_2018['tarih'])
    df_2019['tarih'] = pd.to_datetime(df_2019['tarih'])

    
    # Tarih formatını ay-gün-yıl olarak değiştir
    aralik_missing = aralik
    for day in days:
        aralik = aralik.union(pd.date_range(start=day+" 00:00", end=day+" 23:55", freq="5T"))
    for day in days:
        aralik_missing = aralik_missing.union(pd.date_range(start=pd.to_datetime(day, format='%m-%d-%Y') + pd.DateOffset(days=0), end=pd.to_datetime(day, format='%m-%d-%Y') + pd.DateOffset(days=10), freq="5T"))
      
    # Verileri birleştir ve tatil/bayram sütunlarını ekle
    df = pd.concat([df_2018, df_2019]).reset_index(drop=True)
    df["consec_hol"] = np.zeros(len(df["hiz"]))
    df.set_index("tarih",inplace=True)
    for i,day in enumerate(tatil):
       df.consec_hol.loc[day] = i+1
    for i,day in enumerate(tatil_test):
        df.consec_hol.loc[day] = i+1
    tatil = flatten_comprehension([pd.date_range(start=day+" 00:00", end=day+" 23:55", freq="5T") for day in tatil])
    tatil_test = flatten_comprehension([pd.date_range(start=day+" 00:00", end=day+" 23:55", freq="5T") for day in tatil_test])
    df.reset_index(inplace=True)
    df["bayram"] = df["tarih"].isin(pd.to_datetime(tatil, format='%Y-%m-%d %H:%M:%S')).astype(int)
    # ay gün yıl
    # Tahmin yapılacak tarih aralığını belirle
    # Saat, ay, haftanın günü ve gün bilgilerini ekle
    df["year"] = df["tarih"].dt.year
    df["saat"] = df["tarih"].dt.hour
    df["min"] = df["tarih"].dt.minute
    df["hour_cos"] = np.cos(2 * np.pi * df["tarih"].dt.hour/24.0) + 0.000001
    df["hour_sin"] = np.sin(2 * np.pi * df["tarih"].dt.hour/24.0) + 0.000001
    df["min_cos"] = np.cos(2 * np.pi * df["tarih"].dt.minute/60.0) + 0.000001
    df["min_sin"] = np.cos(2 * np.pi * df["tarih"].dt.minute/60.0) + 0.000001
    df["month"] = df["tarih"].dt.month
    df["week"] = df["tarih"].dt.week
    df["dofweek"] = df["tarih"].dt.dayofweek
    df["day"] = df["tarih"].dt.day
    df["dayofyear"] = df["tarih"].dt.dayofyear
    df['is_weekend'] = np.where(df['dofweek'].isin([5,6]), 1,0)
    
    # Tahmin yapılacak veri kümesini oluştur
    df_tahmin = df.set_index("tarih").loc[aralik_test]
    df_tahmin.reset_index(names="tarih",inplace=True)
    df_tahmin.replace(-1, np.nan, inplace=True)
   
    df_tahmin.dropna(inplace=True,how='any',axis=0)
    df_tahmin["bayram"] = df_tahmin["tarih"].isin(pd.to_datetime(tatil_test, format='%Y-%m-%d %H:%M:%S')).astype(int)
    df_tahmin["dayofyear"] = df_tahmin["tarih"].dt.dayofyear
    df_tahmin["day"] = df_tahmin["tarih"].dt.day
    df_tahmin.set_index("tarih",inplace=True)
    
    df = df.set_index("tarih").loc[aralik_missing]
   
    df.reset_index(names="tarih",inplace=True)
    
   
    df = fill_missing_values(df)
    df = df.set_index("tarih").loc[aralik]
    df.dropna(inplace=True)
    df_DNN = df.copy()
    df["hiz_low"], df["hiz_high"] = low_passFilter(df, file_1)
    df.reset_index(names="tarih",inplace=True)
    df_DNN.reset_index(names="tarih",inplace=True)
    return df, df_tahmin, df_DNN

def train_test_split(df, df_tahmin):
    # Eğitim verilerini seç
    X_train = df[["tarih","bayram","saat","hour_cos","hour_sin","is_weekend","dofweek","month","min","consec_hol"]]
    X_train.set_index("tarih",inplace=True)

    # Eğitim verilerini seç
    X_test = df_tahmin[["bayram","saat","hour_cos","hour_sin","is_weekend","dofweek","month","min","consec_hol"]]
    y_test = df_tahmin["hiz"]
    
    kmeans = KMeans(n_clusters=432, random_state=0, n_init="auto").fit(X_train)
    labels = kmeans.labels_
    kmean_pred = kmeans.predict(X_test)
    centers = kmeans.cluster_centers_
    
    
    if len(df.columns) > 17:
        X_train["Label"] = labels
        X_test["Label"] = kmean_pred
        y_train_low = df["hiz_low"]
        y_train_high = df["hiz_high"]
    else:
        y_train_low = df["hiz"]
        y_train_high = None
        
    # Eğitim verilerini seç
    X_train = (X_train-X_train.min())/(X_train.max()-X_train.min() + 0.000000000000000001) 
    X_test = (X_test-X_test.min())/(X_test.max() - X_test.min()  + 0.000000000000000001) 
    return X_train, y_train_low, y_train_high, X_test, y_test


def fill_missing_values(df):
    # DataFrame'i tarih sırasına göre sırala
    df = df.sort_values(by='tarih')

    # Tüm eksik değerleri -1 ile NaN olarak değiştir
    df.replace(-1, np.nan, inplace=True)
    sayac = 0
    # Eksik değerleri doldur
    for index, row in df.iterrows():
        if pd.isnull(row['hiz']):
            # Eksik değer varsa
            date = row['tarih']
            hour = date.hour

            # Önceki 5 günün aynı saatindeki hızları al
            previous_days = df[(df['tarih'].dt.time == time(hour)) & (df['tarih'] < date) & (~df['hiz'].isnull())].tail(5)

            # Hız ortalamasını hesapla
            if not previous_days.empty:
                avg_speed = previous_days['hiz'].mean()
                #print("Eksik yok: ",avg_speed," ",sayac)
                # Eksik değeri ortalamayla doldur
                df.at[index, 'hiz'] = avg_speed
            else:
                # Önceki günlerde eksik veri varsa, ileri günleri kontrol et
                next_days = df[(df['tarih'].dt.time == time(hour)) & (df['tarih'] > date) & (~df['hiz'].isnull())].head(5)
                if not next_days.empty:
                    avg_speed = next_days['hiz'].mean()

                    # Eksik değeri ortalamayla doldur
                    df.at[index, 'hiz'] = avg_speed
                    print("Eksik var: ",avg_speed)
        sayac+=1

    return df