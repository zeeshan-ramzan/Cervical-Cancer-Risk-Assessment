import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn import metrics  
from sklearn.model_selection import train_test_split  
import math


data = pd.read_csv('Biopsy.csv', low_memory=False)
target = data["Biopsy"]

train_data, test_data, train_target, test_target = train_test_split(data, target, train_size = 0.8)  
train_data.shape  

data["Age"] = np.log((data["Age"] + 0.1).astype(float))
data["Number of sexual partners"] = np.log((data["Number of sexual partners"] + 0.1).astype(float))
data["First sexual intercourse"] = np.log((data["First sexual intercourse"] + 0.1).astype(float))
data["Num of pregnancies"] = np.log((data["Num of pregnancies"] + 0.1).astype(float))
data["Smokes"] = np.log((data["Smokes"] + 0.1).astype(float))
data["Smokes (years)"] = np.log((data["Smokes (years)"] + 0.1).astype(float))

#for row in data:
#	print (row)

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(train_data)



preds = clf.predict(test_data)  
targs = test_target

print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds))  
print("recall: ", metrics.recall_score(targs, preds))  
print("f1: ", metrics.f1_score(targs, preds))  
