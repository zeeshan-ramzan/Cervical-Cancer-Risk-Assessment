
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn import metrics  
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import LocalOutlierFactor 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cluster import KMeans
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
              "IUD",
              "IUD (years)",
              "STDs",
              "STDs (number)",
              "STDs:condylomatosis",
              "STDs:cervical condylomatosis",
              "STDs:vaginal condylomatosis",
              "STDs:vulvo-perineal condylomatosis",
              "STDs:pelvic inflammatory disease",
              "STDs:genital herpes",
              "STDs:molluscum contagiosum",
              "STDs:AIDS",
              "STDs:HIV",
              "STDs:Hepatitis B",
              "STDs:HPV",
              "STDs: Number of diagnosis",
              "STDs: Time since last diagnosis",
              "Dx:Cancer",
              "Dx:CIN",
              "Dx"

]


pdata_train_selected_feature = pdata_train[relevant_features]
full_test_data= full_test_data[relevant_features]


pdata_train_selected_feature = pdata_train_selected_feature[1:23]
full_test_data = full_test_data[1:23]
train_target= test_target[1:23]
test_target = test_target[10:32]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(pdata_train_selected_feature, train_target)

pred=clf.predict(full_test_data)

print(confusion_matrix(test_target,pred))
print(classification_report(test_target,pred))

print(len(clf.coefs_))
print(len(clf.coefs_[0]))
print(len(clf.intercepts_[0]))

print(pred)
print(test_target)
#print(pdata_train_selected_feature)
#kmeans = KMeans(n_clusters=2, random_state=0)
#kmeans.fit(pdata_train_selected_feature)

#kpred = kmeans.predict(full_test_data)

#print(test_target)
#print(kpred)

#pdata_train_selected_feature = pdata_train_selected_feature[1:25]
#full_test_data = full_test_data[1:25]
#plt.scatter(pdata_train_selected_feature,full_test_data,c=kpred, s=50)
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#plt.show()



















clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(pdata_train_selected_feature)

preds = clf.predict(full_test_data) 
targs = test_target

#print("accuracy: ", metrics.accuracy_score(targs, preds))  
#print("precision: ", metrics.precision_score(targs, preds, average='macro'))  
#print("recall: ", metrics.recall_score(targs, preds, average='macro'))  
#print("f1: ", metrics.f1_score(targs, preds, average='macro'))    



#model = LocalOutlierFactor(n_neighbors=20)
#y_pred = model.fit_predict(full_test_data)
#y_pred_outliers = y_pred[200:]


