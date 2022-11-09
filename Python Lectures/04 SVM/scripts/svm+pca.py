import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


XX = df_colon.drop('Class', axis = 1)
yy = df_colon['Class']

X, y = smote.fit_resample(XX, yy)
X, y = X.to_numpy(), y.to_numpy()
y = y.flatten()

plot_data(X, y)

random.seed(413)
pca = PCA(n_components = 2)
pca.fit(X)
X_pca = pca.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.20)

tuned_parameters = [{'kernel': ['rbf'],     'gamma': [1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7], 'C': [0.001, 0.10, 0.1, 10, 20, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7], 'C': [0.001, 0.10, 0.1, 10, 20, 25, 50, 100, 1000]},
                    {'kernel': ['linear'],                                              'C': [0.001, 0.10, 0.1, 10, 20, 25, 50, 100, 1000]}]

scoring = {
    'Precision': 'precision',
    'Recall': 'recall',
    'Accuracy': 'accuracy',
    'AUC': 'roc_auc',
    'F1': 'f1_micro'}
    
random.seed(413)
clf = GridSearchCV(
              SVC(), param_grid = tuned_parameters,
              scoring = scoring, refit = 'F1',
              return_train_score = True)
clf.fit(X_train, y_train)
results = clf.cv_results_

print('Best parameters set found on development set:')
print(clf.best_params_)

random.seed(413)
svclassifier = SVC(kernel = 'rbf', C = 1000, gamma = 0.01)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))