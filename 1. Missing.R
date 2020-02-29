library(e1071)
library(mlbench)
library(caret)
library(Amelia)

# uploading data
data<-read.table(file.choose(),header=T,sep=",")

# To Check
is.na(data)	
missmap(data)

# To Remove
na.omit(data)
