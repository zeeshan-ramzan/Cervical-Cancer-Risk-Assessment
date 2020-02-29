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

testing_data = pd.read_csv('Biopsy_Test.csv',low_memory=False)

relevant_features = [  
              
              "Dx:CIN",
              "Dx"

]


data = data[relevant_features]

testing_data = testing_data[relevant_features]

train_data, test_data, train_target, test_target = train_test_split(data, target, train_size = 0.8)  
train_data.shape  

#data["Age"] = np.log((data["Age"] + 0.1).astype(float))
#data["Number of sexual partners"] = np.log((data["Number of sexual partners"] + 0.1).astype(float))
#data["First sexual intercourse"] = np.log((data["First sexual intercourse"] + 0.1).astype(float))
#data["Num of pregnancies"] = np.log((data["Num of pregnancies"] + 0.1).astype(float))
#data["Smokes"] = np.log((data["Smokes"] + 0.1).astype(float))
#data["Smokes (years)"] = np.log((data["Smokes (years)"] + 0.1).astype(float))

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

pred_train = clf.predict(train_data)
pred_test = clf.predict(test_data)
error_train = pred_train[pred_train == -1].size
error_test = pred_test[pred_test == -1].size

X_outliers = testing_data[testing_data.Dx==0]
pred_outliers = clf.predict(X_outliers)
error_outliers = pred_outliers[pred_outliers == 1].size


xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(train_data, train_data, c='white', s=s, edgecolors='k')
b2 = plt.scatter(test_data, test_data, c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers, X_outliers, c='gold', s=s,
                edgecolors='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (error_train, error_test, error_outliers))
plt.show()