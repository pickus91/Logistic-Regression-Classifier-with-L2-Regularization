# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:10:54 2017

@author: pickus
"""
#Logistic Regression from Scratch
import numpy as np




class LogisticRegression(object):    
    """Logistic Regression for binary classification"""
    
    def __init__(self, learningRate, numIterations, penalty):
        
        self.learningRate = learningRate
        self.numIterations = numIterations
        self.penalty = penalty
        
    def train(self, X_train, y_train, C = 0.1, tol = 10 ** -4):
        
        tolerance = tol * np.ones([1, np.shape(X_train)[1] + 1])
        self.weights = np.zeros(np.shape(X_train)[1] + 1) 
        X_train = np.c_[np.ones([np.shape(X_train)[0], 1]), X_train]
        self.costs = []

        for i in range(self.numIterations):
            
            z = np.dot(X_train, self.weights)
            errors = y_train - logistic_func(z)
            if self.penalty is not None:                
                delta_w = self.learningRate * (C * np.dot(errors, X_train) + np.sum(self.weights))  
            else:
                delta_w = self.learningRate * np.dot(errors, X_train)
                #what you had before
            self.iterationsPerformed = i

            if np.all(abs(delta_w) >= tolerance): 
                #weight update
                self.weights += delta_w                                
                #Costs
                if self.penalty is not None:
                    self.costs.append(reg_logLiklihood(X_train, self.weights, y_train, C))
                else:
                    self.costs.append(logLiklihood(z, y_train))
            else:
                break
            
        return self
                    
    def predict(self, X_test, pi):
        
        z = self.weights[0] + np.dot(X_test, self.weights[1:])        
        probs = np.array([logistic_func(i) for i in z])
        predictions = np.where(probs >= pi, 1, 0)
       
        return predictions, probs
        
    def performanceEval(self, predictions, y_test):
        #Initialize
        TP, TN, FP, FN, P, N = 0, 0, 0, 0, 0, 0
        
        for idx, test_sample in enumerate(y_test):
            
            if predictions[idx] == 1 and test_sample == 1:
                TP += 1       
                P += 1
            elif predictions[idx] == 0 and test_sample == 0:                
                TN += 1
                N += 1
            elif predictions[idx] == 0 and test_sample == 1:
                FN += 1
                P += 1
            elif predictions[idx] == 1 and test_sample == 0:
                FP += 1
                N += 1
            
        #Accuracy
        accuracy = (TP + TN) / (P + N)                
        #Sensitivity (TPR) or "recall"
        sensitivity = TP / P        
        #Specificiy (TNR)
        specificity = TN / N        
        #Positivie Predictive Value (PPV) or "precision"
        PPV = TP / (TP + FP)        
        #Negative Predictive Value (NPV)
        NPV = TN / (TN + FN)        
        #False Negative Rate (FNR) or "miss rate"
        FNR = 1 - sensitivity        
        #False Positive Rate (FPR) or "fall out"
        FPR = 1 - specificity
        
        performance = {'Accuracy': accuracy, 'Sensitivity': sensitivity,
                       'Specificity': specificity, 'Precision': PPV,
                       'NPV': NPV, 'FNR': FNR, 'FPR': FPR}        
      
        return performance
        
    def predictionPlot(self, X_test, y_test):
      
        zs = self.weights[0] + np.dot(X_test, self.weights[1:])        
        probs = np.array([logistic_func(i) for i in zs])
        
        plt.figure()
        plt.plot(np.arange(-10, 10, 0.1), logistic_func(np.arange(-10, 10, 0.1)))
        
        colors = ['r','g','b','c']
        cmap = ListedColormap(colors[:len(np.unique(y_test))])
        probs = np.array(probs)
        for idx,cl in enumerate(np.unique(y_test)):
            plt.scatter(x = zs[np.where(y_test == cl)[0]], y = probs[np.where(y_test == cl)[0]],
                    alpha = 0.8, c = cmap(idx),
                    marker = 'o', label = cl, s = 30)

        plt.xlabel('z')
        plt.ylim([-0.1, 1.1])
        plt.axhline(0.0, ls = 'dotted', color = 'k')
        plt.axhline(1.0, ls = 'dotted', color = 'k')
        plt.axvline(0.0, ls = 'dotted', color = 'k')
        plt.ylabel('$\phi (z)$')
        plt.legend(loc = 'upper left')
        plt.title('Logistic Regression Prediction Curve')
        plt.show()
        
    def plotCost(self):
        
        plt.figure()
        plt.plot(np.arange(1, self.iterationsPerformed + 1), self.costs, marker = '.')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Liklihood J(w)')
        
        
    def plotDecisionRegions(self, X_test, y_test, pi = 0.5, res = 0.01):
        
        x = np.arange(min(X_test[:,0]) - 1, max(X_test[:,0]) + 1, 0.01)
        y = np.arange(min(X_test[:,1]) - 1, max(X_test[:,1]) + 1, 0.01)        
        xx, yy = np.meshgrid(x, y, indexing = 'xy')
        
        data_points = np.transpose([xx.ravel(), yy.ravel()])
        preds, probs = self.predict(data_points, pi)
            
        colors = ['r','g','b']
        cmap = ListedColormap(colors[:len(np.unique(y_test))])
        probs = np.array(probs)
                
        for idx,cl in enumerate(np.unique(y_test)):
            plt.scatter(x = X_test[:,0][np.where(y_test == cl)[0]], y = X_test[:,1][np.where(y_test == cl)[0]],
                    alpha = 0.8, c = cmap(idx),
                    marker = 'o', label = cl, s = 30)
                    
        preds = preds.reshape(xx.shape)
        plt.contourf(xx, yy, preds, alpha = 0.3)
        plt.legend(loc = 'best')
        plt.xlabel('$x_1$', size = 'x-large')
        plt.ylabel('$x_2$', size = 'x-large')

def logistic_func(z):   
    return 1 / (1 + np.exp(-z))  
    
def logLiklihood(z, y):
    """Regularized cost function to be minimized"""
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z))))
    
def reg_logLiklihood(x, weights, y, C):
    
    """Regularized cost function to be minimized"""    
    z = np.dot(x, weights) #Assume first column of x is 1's...
    reg_term = (1 / (2 * C)) * np.dot(weights.T, weights)
    
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z)))) + reg_term

#%%
#Import data
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import ListedColormap
style.use('ggplot')

#UCI
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header = None)

X = df.loc[:, 1:10]
y = df.loc[:, 10] #labels 

le = LabelEncoder()
y = le.fit_transform(y)
X = X.replace('?', np.nan)

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr = imr.fit(X)
X_imputed = imr.transform(X.values)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size = 0.3) #random_state = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

eta = 0.01
numIters = 30
LR = LogisticRegression(eta, numIters, penalty = 'L2')  
LR.train(X_train_pca, y_train, C = 0.01, tol = 10 ** -3)

predictions, probs = LR.predict(X_test_pca, 0.5)
performance = LR.performanceEval(predictions, y_test)
LR.plotDecisionRegions(X_test_pca, y_test)
#%% Varying C
param_range = np.logspace(-5, 4, 20)

accuracy = []
weights_PC1 = []
weights_PC2 = []
for c in param_range:
    LR.train_L2(X_train_pca, y_train, c)
    predictions, probs = LR.predict(X_test_pca, 0.5)
    performance = LR.performanceEval(predictions, y_test)
    accuracy.append(performance['Accuracy'])
    weights_PC1.append(LR.weights[1])
    weights_PC2.append(LR.weights[2])
    
plt.figure()
plt.plot(param_range, accuracy, marker = '.')
plt.grid()
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Accuracy')

#Plotting how weights go down to zero!
plt.figure()
plt.plot(param_range, weights_PC1, ls = '--', label = 'PC1')
plt.plot(param_range, weights_PC2, label = 'PC2')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Weight Coefficients')
plt.legend(loc = 'best')

