import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn import metrics  
from sklearn.model_selection import train_test_split  
import math


data = pd.read_csv('Citology_Test.csv', low_memory=False)
target = data["Citology"]

testing_data = pd.read_csv('Citology.csv',low_memory=False)
#target = testing_data["Citology"]

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


data = data[relevant_features]


train_data, test_data, train_target, test_target = train_test_split(data, target, train_size = 0.8)  
train_data.shape  


#for row in data:
#	print (row)

clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
clf.fit(data,target)

preds = clf.predict(test_data)  
targs = test_target

print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds, average='macro'))  
print("recall: ", metrics.recall_score(targs, preds, average='macro'))  
print("f1: ", metrics.f1_score(targs, preds, average='macro'))  