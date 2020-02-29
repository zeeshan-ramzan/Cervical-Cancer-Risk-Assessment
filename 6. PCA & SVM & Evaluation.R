library(e1071)
library(mlbench)
library(caret)

# uploading data
data<-read.table(file.choose(),header=T,sep=",")

# dividing data into testdata and traindata
data2 <- sample(2,nrow(data),replace = TRUE,prob = c(0.70,0.30))
traindata <- data[data2==1,]
testdata <- data[data2==2,]
#traindata <- traindata[1:230, ]
#testdata <- testdata[1:230, ]

# PCA
prin_comp <- prcomp(traindata, scale. = T)
names(prin_comp)

summary(prin_comp)
str(prin_comp)

# install_github("vqv/ggbiplot")
library(devtools)
library(ggbiplot)
ggbiplot(prin_comp, scale = 0)

# PCA Properties
prin_comp$center
prin_comp$scale
prin_comp$rotation
prin_comp$rotation[1:5,]


std_dev <- prin_comp$sdev
pr_var <- std_dev^2
pr_var[1:9]
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:9]
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

# SVM
library(e1071)
library(mlbench)
library(caret)
model_svm <- svm(traindata)
# model_svm <- svm(traindata ,y=NULL, type='one-classification', nu=0.5, scale=TRUE, kernel="radial")

# Evaluation
pred <- predict(model_svm, testdata)
#cf <- confusionMatrix(pred,testdata$Outcome)
cf<-table(Predicted=pred,Reference=testdata$Outcome)
cf

# Error
error <- traindata$Outcome - pred
svm_error <- sqrt(mean(error^2))
svm_error

