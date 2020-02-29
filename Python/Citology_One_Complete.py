
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn import metrics  
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import LocalOutlierFactor 
from sklearn.metrics import confusion_matrix
import math


data = pd.read_csv('Complete_Citology.csv', low_memory=False)


pdata = data[data.Citology == 1]
ndata = data[data.Citology == 0]

p_rnd = np.random.rand(len(pdata)) > 0.5
n_rnd = np.random.rand(len(ndata)) > 0.05


pdata_train = pdata[p_rnd]
pdata_Test = pdata[~p_rnd]


ndata_Train = ndata[n_rnd]
ndata_Test = ndata[~n_rnd]

frames = [pdata_Test,ndata_Test]

full_test_data = pd.concat(frames)
test_target = full_test_data["Citology"]

relevant_features = [  
"Hormonal Contraceptives (years)",
"STDs:vulvo-perineal condylomatosis",
"STDs:syphilis",
"STDs:pelvic inflammatory disease",
"STDs:genital herpes",
"STDs:molluscum contagiosum",
"STDs:HIV",
"Dx:Cancer",
"Dx:HPV",
"Dx"

]


pdata_train_selected_feature = pdata_train[relevant_features]
full_test_data= full_test_data[relevant_features]

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(pdata_train_selected_feature)

print(len(pdata_train_selected_feature))

preds = clf.predict(full_test_data) 
targs = test_target

print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds, average='macro'))  
print("recall: ", metrics.recall_score(targs, preds, average='macro'))  
print("f1: ", metrics.f1_score(targs, preds, average='macro'))    

print(confusion_matrix(targs, preds))

'''
model = LocalOutlierFactor(n_neighbors=20)
y_pred = model.fit_predict(full_test_data)
y_pred_outliers = y_pred[200:]


print(len(test_target))
print(len(y_pred))
print(preds)
print(test_target)
'''



