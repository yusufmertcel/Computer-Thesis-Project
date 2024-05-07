from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.model_selection import GridSearchCV
import os

# Tahminleri almak ve hata oranını hesaplamak için fonksiyon
def get_pred(model, X_train, y_train, X_test, y_test):
    # Modeli eğit
    model.fit(X_train, y_train)

    # Tahminleri yap
    y_pred = model.predict(X_test.values)
    
    # Hata oranını hesaplama
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return y_pred, mape

# Model tahminlerini görselleştirmek için bir fonksiyon
def plot_model(plot_name, df_tahmin, path_dir, segment_name):
    df_tahmin.columns = ["hiz_tahmin","hiz"]
    # Bir grafik oluştur
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Gerçek ve tahmin edilen hızı çiz
    ax.plot(df_tahmin.index, df_tahmin["hiz"], label="Gerçek Değer")
    ax.plot(df_tahmin.index, df_tahmin["hiz_tahmin"], label="Tahmin")
    
    # Eksen etiketleri ve başlık ayarla
    ax.set_xlabel("Saat")
    ax.set_ylabel("Hız")
    ax.set_title(plot_name+segment_name)
    
    
    ax.legend()
    if not os.path.exists(f"{path_dir}/{segment_name}"):
        os.mkdir(f"{path_dir}/{segment_name}")
    # Grafiği göster
    plt.savefig(f"D:/CE 3.sınıf/Spring Semester/Ara Proje/Results/{path_dir}/{segment_name}/{segment_name}{plot_name}.png")
    plt.show()
    return fig

# En iyi ve en kötü tahminleri analiz etmek için bir fonksiyon
def best_and_worst(y_test,y_pred):
    # Tahmin ve gerçek hız değerlerini içeren bir DataFrame oluştur
    errors = pd.DataFrame(y_test)
    errors["preds"] = y_pred
    
    # Hata ve mutlak hata değerlerini hesapla
    errors['error'] = errors['hiz'] - y_pred
    errors['abs_error'] = errors['error'].apply(np.abs)
    
    # Hata ve mutlak hata değerlerini hesapla
    errors['year'] = y_test.index.year
    errors['month'] = y_test.index.month
    errors['dayofmonth'] = y_test.index.day
    
    # Günlük ortalama hız, tahmin, hata ve mutlak hata değerlerini hesapla
    error_by_day = errors.groupby(['year','month','dayofmonth']) \
        .mean()[['hiz','preds','error','abs_error']]
    return error_by_day