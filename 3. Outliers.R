# uploading data
data<-read.table(file.choose(),header=T,sep=",")
mod <- lm(Outcome ~ ., data=data)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=1.5, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels


