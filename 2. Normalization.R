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

# Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
traindata <- normalize(traindata[,1:8])
testdata <- normalize(testdata[,1:8])
