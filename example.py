# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:35:34 2017

@author: pickus

Classification of data in the Breast Cancer Wisconsin dataset using the
LogisticRegression class with L2 regularization. More detailed information
on this dataset can be found at: 

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from logisticRegressionClassifier import LogisticRegression

##Import Breast Cancer Wisconsin dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header = None)

X = df.loc[:, 1:10] #features vectors
y = df.loc[:, 10]   #class labels: 2 = benign, 4 = malignant

le = LabelEncoder() #positive class = 1 (benign), negative class = 0 (malignant)
y = le.fit_transform(y)

#Replace missing feature values with mean feature value
X = X.replace('?', np.nan)
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr = imr.fit(X)
X_imputed = imr.transform(X.values)

#Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size = 0.3, random_state = 1)

#Z-score normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Principle component analysis (dimensionality reduction)
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#Training logistic regression classifier with L2 penalty
LR = LogisticRegression(learningRate = 0.01, numIterations = 20, penalty = 'L2', C = 0.01)  
LR.train(X_train_pca, y_train, tol = 10 ** -3)

#Testing fitted model on test data with cutoff probability 50%
predictions, probs = LR.predict(X_test_pca, 0.5)
performance = LR.performanceEval(predictions, y_test)
LR.plotDecisionRegions(X_test_pca, y_test)
LR.predictionPlot(X_test_pca, y_test)

#Print out performance values
for key, value in performance.items():
    print('%s : %.2f' % (key, value))


