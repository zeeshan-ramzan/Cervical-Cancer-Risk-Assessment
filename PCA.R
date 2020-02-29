  data1 <- read.table(file.choose(),header=T,sep=",")
  arc.pca1 <- princomp(data1, score=TRUE, cor=TRUE)
  summary(arc.pca1)
  plot(arc.pca1)
  biplot(arc.pca1)
  screeplot(arc.pca1,type="line",main="Plot")
  arc.pca1$scores
  
  
  
 data2 <-cbind(data1$Pregnancies,data1$Glucose,data1$SkinThickness,data1$Outcome)
 arc.pca1 <- princomp(data2, score=TRUE, cor=TRUE)
  summary(arc.pca1)
 plot(arc.pca1)
  biplot(arc.pca1)
 screeplot(arc.pca1,type="line",main="Plot")
  arc.pca1$scores
  