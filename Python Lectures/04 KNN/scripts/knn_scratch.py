"""
Created on Fri Nov 04 12:46:39 2022
@author: Romen Samuel Wabina
"""

import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def find_distance(self, X_train, X_test):
        #create newaxis simply so that broadcast to all values
        dist = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dist = dist ** 2

        summed_dist = sq_dist.sum(axis=2)
        sq_dist = np.sqrt(summed_dist)
        return sq_dist
    
    def find_neighbors(self, X_train, X_test):
        dist = self.find_distance(X_train, X_test)
        #return the first k neighbors
        neighbors_ix = np.argsort(dist)[:, 0:self.k]
        return neighbors_ix
    
    def find_neighbors_without_k(self, X_train, X_test):
        dist = self.find_distance(X_train, X_test)
        #return the first k neighbors
        neighbors_ix = np.argsort(dist)
        return neighbors_ix
    
    def get_most_common(self, y, n_classes, X_train, X_test):
        y_nearest = y[0:self.k]
        bincount = np.bincount(y_nearest, minlength=n_classes)
        largest = bincount.argmax()
        second_largest = bincount.argsort()[-2:][0]
        
        #if the first two most common is the same, we take the third most common as the decider
        if bincount[largest] == bincount[second_largest]:
            y_nearest = y[0: self.k+1]  #add one more neighbor
            return np.bincount(y_nearest).argmax(), bincount[largest] / bincount.sum()
        return np.bincount(y_nearest).argmax(), bincount[largest] / bincount.sum()
        
    def cv(self, X_train, y_train, cv, k):
        foldsize = int(X_train.shape[0]/cv)
        yhat_cv = np.zeros((len(k), cv))
        yhat_cv_prob = np.zeros((len(k), cv))
        
        for k_idx, kneighbors in enumerate(k):
            self.k = kneighbors
            for fold_idx, i in enumerate(range(0, X_train.shape[0], foldsize)):
                X_test_ = X_train[i:i+foldsize]
                y_test_ = y_train[i:i+foldsize]

                X_train_ = np.concatenate((X_train[:i], X_train[i+foldsize:]))
                y_train_ = np.concatenate((y_train[:i], y_train[i+foldsize:]))

                yhat, yhat_prob = self.predict(X_train_, X_test_, y_train_)
                accuracy = np.sum(yhat == y_test_)/len(y_test_)
                yhat_cv[k_idx, fold_idx] = accuracy
                yhat_cv_prob[k_idx, fold_idx] = yhat_prob.mean()
        return yhat_cv, yhat_cv_prob
        
    def predict(self, X_train, X_test, y_train):
        n_classes = len(np.unique(y_train))
        neighbors_ix = self.find_neighbors_without_k(X_train, X_test)
        yhat = np.zeros(X_test.shape[0])
        yhat_prob = np.zeros(X_test.shape[0])
        for ix, y in enumerate(y_train[neighbors_ix]):
            yhat[ix], yhat_prob[ix] = self.get_most_common(y, n_classes, X_train, X_test)
        return yhat, yhat_prob