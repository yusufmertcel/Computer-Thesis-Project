# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:02:05 2023

@author: sceli
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def get_feature_importance(model, feature_names, X_test, y_test, path_dir, segment_name):
    start_time = time.time()
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    if not os.path.exists(f"{path_dir}/{segment_name}"):
        os.mkdir(f"{path_dir}/{segment_name}")
    plt.savefig(f"D:/CE 3.sınıf/Spring Semester/Ara Proje/Results/{path_dir}/{segment_name}/{segment_name}MDI.png")

    start_time = time.time()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig(f"D:/CE 3.sınıf/Spring Semester/Ara Proje/Results/{path_dir}/{segment_name}/{segment_name}permutation.png")
    plt.show()