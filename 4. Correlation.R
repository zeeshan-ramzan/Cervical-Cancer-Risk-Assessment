data<-read.table(file.choose(),header=T,sep=",")
library(corrplot)
m <- cor(data.matrix(data[,c(1,2,3,4,5,6,7,8)]),data.matrix(data), use="complete.obs", method="spearman")
corrplot(m, method = "number")
