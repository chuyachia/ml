## The following codes implement an adaptative boosting algorithm for 
## binary classification that uses decision stumps as the base algorithm.
## The iris dataset is used as an example here.

## load data
iris_bi <- iris
iris_bi$Species <- ifelse(iris$Species=="virginica",1,-1)

## seperate train and test
train_indx<-sample(150,100)
X <- iris_bi[train_indx,c(1:4)]
Y <- iris_bi[train_indx,5]
n <- nrow(X)
f <- ncol(X)

## parameters
# number of models to generate
nMod <- 20

## initialize to uniform vote
w <- rep(1/n,n)
vote_df <- NULL
predict_df <- NULL
dir_df <- NULL
thres_df <- NULL
feature_df <- NULL
for (j in 1 : nMod)
{
  thres <- rep(NA,f)
  dir<-  rep(NA,f)
  e <-  rep(NA,f)
  predict <- matrix(nrow=n,ncol=f)
  ## train one decision stump on each feature
  for (i in 1 :f)
  {
    indx <- order(X[,i])
    x_order <- X[indx,i]
    y_order <- Y[indx]* w[indx]
    data_order <- aggregate(y_order~x_order,FUN=sum)
    max_ind <-which.max(abs(cumsum(data_order$y_order)))
    thres[i] <- data_order$x_order[max_ind]
    dir[i] <- -sign(cumsum(data_order$y_order)[max_ind])
    predict[,i] <-  dir[i]*ifelse((X[,i]-thres[i])>0,1,-1)
    e[i] <-  sum((Y!=predict[,i])*w)/sum(w)
  }
  ## choose the best feature to use
  feature <- which.min(e)
  thres_df <- rbind(thres_df,thres[feature])
  dir_df <- rbind(dir_df,dir[feature])
  feature_df<- rbind(feature_df,feature)
  ## predict
  P <- predict[,feature]
  E <- e[feature]
  S <- sqrt((1-E)/E)
  ## update weights (so that the next model will give different results)
  w[which(Y!=P)] <- w[which(Y!=P)]*S
  w[which(Y==P)] <- w[which(Y==P)]/S
  vote <- log(S)
  predict_df<- cbind(predict_df,P)
  vote_df <- rbind(vote_df,vote)
}

## In-sample error
length(which(sign(predict_df %*% vote_df)!=Y))/length(Y)

## out-of-sampe error
X_test <- iris_bi[-train_indx,c(1:4)]
Y_test <- iris_bi[-train_indx,5]
X_expand <- sapply(feature_df,function(x)X_test[,x])
X_expand<-sapply(1:ncol(X_expand),function(x){
  dir_df[x]*ifelse((X_expand[,x]-thres_df[x])>0,1,-1)})

length(which(sign(X_expand%*%vote_df)!=Y_test))/length(Y_test)

