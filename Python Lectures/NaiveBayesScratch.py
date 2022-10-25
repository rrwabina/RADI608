import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report
import numpy as np
import pandas as pd

##################### Given Problem ###############################
n = 50
x1_feature = np.random.choice(3, n, p=[0.2, 0.1, 0.7])
x2_feature = np.random.randn(n )*1.5 + 6
y = np.zeros(n)
class0_features = np.array([x1_feature, x2_feature, y]).T

x1_feature = np.random.choice(3, n, p=[0.2, 0.5, 0.3])
x2_feature = np.random.randn(n )*3 + 2
y = np.ones(n)
class1_features = np.array([x1_feature, x2_feature, y]).T

data  = np.concatenate([class0_features, class1_features])

data = pd.DataFrame(data, columns = ['features1', 'features2', 'y'])
####################################################################


X = data[['features1', 'features2']].to_numpy()
y = data['y'].to_numpy()

plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, s = 25, edgecolor = 'k')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Get the mean and standard deviation of every row in dataset 
def mean_std(X_train, y_train):
    n = X_train.shape[1]
    mean = np.zeros((2, n))
    std = np.zeros((2, n))
    for label in [0, 1]:
        mean[label, :] = X_train[y_train==label].mean(axis=0)
        std[label, :]  = X_train[y_train==label].std(axis=0)
    return mean, std

# Get the mean and standard deviation of the training dataset 
mean, std = mean_std(X_train, y_train)

# Fitting the data with the Gaussian distribution function 
def gaussian_pdf(X, mean, std):
    left = 1 / (np.sqrt(2 * np.pi) * std)
    e = (X - mean) ** 2 / (2 * (std ** 2))
    right = np.exp(-e)
    return left*right

# Likelihood of every output y = 0 and y = 1
likelihood0 = gaussian_pdf(X_test, mean[0, :], std[0, :])
likelihood1 = gaussian_pdf(X_test, mean[1, :], std[0, :])

total_likelihood0 = np.prod(likelihood0, axis = 1)
total_likelihood1 = np.prod(likelihood1, axis = 1)

m0 = len(X_train[y_train == 0])
m1 = len(X_train[y_train == 1])
prior0 = m0 / (m0 + m1)
prior1 = m1 / (m0 + m1)

posterior0 = prior0 * total_likelihood0    
posterior1 = prior1 * total_likelihood1
yhat = 1 * posterior1 > posterior0


print(average_precision_score(y_test, yhat))
print("Report: ", classification_report(y_test, yhat))