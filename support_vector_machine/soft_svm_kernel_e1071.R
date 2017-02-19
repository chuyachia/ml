## Load dta
library(dplyr)
bcancer <- read.table("breast_cancer.data",sep=",",stringsAsFactors = F)
bcancer$V7 <- as.numeric(bcancer$V7)
bcancer <- bcancer[-which(is.na(bcancer$V7)),] #V7 contains some NA data
bcancer <- bcancer %>% select(-V1)# V1 is the sample code number

## Plot
library(gridExtra)
library(ggplot2)
drawplot <- function(x){ggplot(bcancer,aes(bcancer[,x],fill=as.factor(V11)))+
    geom_bar(alpha = 0.5, position = 'identity')+
    scale_fill_manual(values  = c("green","red"),name="",guide=FALSE) +
    xlab("")+ggtitle(x)}

for (i in 1:9){
  local({
    j <-i 
    assign(paste0('p', i), drawplot(colnames(bcancer)[j]), pos = .GlobalEnv)
  })  
}
grid.arrange(p1, p2,p3,p4,p5,p6,p7,p8,p9,ncol=3)

## Label data, separate into train and test
x <- bcancer %>% select(-V11)
y <- ifelse(bcancer$V11==4,1,-1)
set.seed(4)
train <- sample(nrow(bcancer),300,replace=F)

## SVM
library(e1071)
tune.svm(x[train,],y[train],gamma=10^(-6:-1),cost=10^(1:4))
# Model without weight adjustment
model <- svm(x=x[train,],y=y[train],scale=FALSE,kernel="radial",
             type= "C-classification",gamma=0.01,cost=10)
y_in <-predict(model,x[train,])
length(which(y_in==y[train]))/length(y[train]) # in-sample accuracy

y_pred <- predict(model,x[-train,])
length(which(y_pred==y[-train]))/length(y[-train]) # out-of-sample accuracy

table(predict=y_pred,diagnosis=y[-train])

# Model with weight adjustment
model <- svm(x=x[train,],y=y[train],scale=FALSE,kernel="radial",
             type= "C-classification",gamma=0.01,cost=10 ,class.weights = c("1"=0.9,"-1"=0.1))
y_pred <- predict(model,x[-train,])
length(which(y_pred==y[-train]))/length(y[-train])
table(predict=y_pred,diagnosis=y[-train])