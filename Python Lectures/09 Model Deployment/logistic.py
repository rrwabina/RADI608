import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

import dataset as data
import warnings 
warnings.filterwarnings('ignore')

X, y = data.get_data()

def scale_data(X, y, test_size = 0.3):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    return X_train, X_test, y_train, y_test

def create_intercepts(X_train, X_test):
    intercept = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((intercept, X_train), axis=1)  
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1)
    return X_train, X_test 
 
def logistic(X_train, X_test, y_train):
    model = LogisticRegression(multi_class = 'ovr')  
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    return yhat

def display_results(yhat, y_test):
    y_test_binarized = label_binarize(y_test, classes = [0, 1, 2])
    yhat_binarized   = label_binarize(yhat,   classes = [0, 1, 2])

    n_classes = len(np.unique(y_test))
    for i in range(n_classes):
        class_score = average_precision_score(y_test_binarized[:, i], yhat_binarized[:, i])
        print(f"Class {i} score: ", np.round(class_score, 3))
    print("Report: ", classification_report(y_test, yhat))