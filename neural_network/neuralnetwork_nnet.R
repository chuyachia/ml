library(dplyr)
library(plotly)
library(reshape2)
library(RColorBrewer)
library(ggplot2)
data <- read.table("glass.data",dec=".",sep=",")
names(data) <- c("ID","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type")
data$Type <- as.factor(data$Type)

library(nnet)
X <- data %>% select(-c(ID,RI,Type)) %>% 
  apply(2,FUN=function(column){return((column-mean(column))/sd(column))} )
Y <- class.ind(data$Type)

## Parameter
fold <- 10
val_s <- floor(nrow(data)/fold)
val_permut <- sample(nrow(data),nrow(data),replace=F)
val <- seq(1,214,by=val_s)
lambda <-seq(0,2,by=0.01)

## Initialize vectors
Ein <- NULL
Ecv <- NULL
Ein_temp <- NULL
Ecv_temp <- NULL

## Cross validation
for (i in 1:length(lambda)){
  for (j in 1: fold){
    if (j < 10)
    {val_index <-val_permut[val[j]:(val[j+1]-1)] }
    else
    {val_index <- val_permut[val[j]:nrow(X)]}
    set.seed(2)
    nnglass <- nnet(X[-val_index,], Y[-val_index,], size =10, rang = 0.1,
                    decay =lambda[i], maxit = 1000,entropy = T,trace=F)
    # In-sample error
    y_pred<- predict(nnglass,X[-val_index,])
    y_pred <- apply(y_pred,1,function(row){colnames(y_pred)[which.max(row)]})
    Ein_temp[j] <- length(which(data$Type[-val_index]!= y_pred))/length(data$Type[-val_index])
    # Cross-validation error
    y_pred<- predict(nnglass,X[val_index,])
    y_pred <- apply(y_pred,1,function(row){colnames(y_pred)[which.max(row)]})
    Ecv_temp[j] <-length(which(data$Type[val_index]!= y_pred))/length(data$Type[val_index])
  }
  Ein[i] <- mean(Ein_temp)  
  Ecv[i] <-mean(Ecv_temp)
}

nnglass <- nnet(X, Y, size =10, rang = 0.1,decay =lambda[which.min(Ecv)], maxit = 1000,entropy = T,trace=F)
min(Ecv)
