
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(required = ['thal', 'cp', 'slope', 'exang', 'label']):
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

    df = pd.read_csv("data/processed.cleveland.data", header = None, names = column_names)

    for column in column_names:
        df = df[(df[column] != "?")].reset_index(drop=True)

    obj_df = df.select_dtypes(include=['object']).copy()

    for column in column_names:
        df[column] = df[column].astype(float)

    df['label'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    df.drop('num', axis=1, inplace = True)

    df = df[required] 
    
    X = df.drop('label', axis = 1).to_numpy()
    y = df['label'].to_numpy().reshape(-1, 1)
    return X, y