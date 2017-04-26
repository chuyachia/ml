## The following codes implement an adaptative boosting algorithm for 
## binary classification that uses decision stumps as the base algorithm.
## The iris dataset is used as an example here.

#### Load data
data(iris)
iris$Species <- ifelse(iris$Species=="versicolor",1,-1)

#### Seperate train and test
train_indx<-sample(150,100)
X_train <- iris[train_indx,c(1:4)]
Y_train <- iris[train_indx,5]
X_test <- iris[-train_indx,c(1:4)]
Y_test <- iris[-train_indx,5]

#### Define functions
train <- function(X,Y,W){
  f <- ncol(X)
  n <- nrow(X)
  thres <- rep(NA,f)
  dir<-  rep(NA,f)
  e <-  rep(NA,f)
  fit <- matrix(nrow=n,ncol=f)
  for (i in 1 :f)
  {
    indx <- order(X[,i])
    x_order <- X[,i]
    y_order <- Y* W
    data_order <- aggregate(y_order~x_order,FUN=sum)
    max_ind <-which.max(abs(cumsum(data_order$y_order)))
    thres[i] <- data_order$x_order[max_ind]
    dir[i] <- -sign(cumsum(data_order$y_order)[max_ind])
    fit[,i] <-  dir[i]*ifelse((X[,i]-thres[i])>0,1,-1)
    e[i] <-  sum((Y!=fit[,i])*W)/sum(W)
  }
  feature <- which.min(e)
  model <- list(feature=feature, 
                threshold=thres[feature],
                direction= dir[feature],
                fit = fit[,feature],
                error= e[feature])
  return(model)
}

predict <- function(model,X){
  model$direction*ifelse((X[,mod$feature]-mod$threshold)>0,1,-1)
}

#### Adaboost
fit_df <- NULL
predict_df <- NULL
vote_df <- NULL
w <- rep(1/length(Y_train),length(Y_train))
nMod <- 20
j <- 0

while(j <nMod) {
## train
mod <- train(X=X_train,Y=Y_train,W=w)

scale <-sqrt((1-mod$error)/mod$error)
vote <- log(scale)
vote_df <- rbind(vote_df,vote)
fit_df <- rbind(fit_df,mod$fit)
## predict
pred <- predict(model=mod,X= X_test)
predict_df <- rbind(predict_df,pred)
## update
w[which(Y_train!=mod$fit)] <- w[which(Y_train!=mod$fit)]*scale
w[which(Y_train==mod$fit)] <- w[which(Y_train==mod$fit)]/scale
j <- j+1
if(mod$error==0) break
}
## aggregate
agg_fit <- t(fit_df)%*%vote_df
agg_predict <- t(predict_df)%*%vote_df
## test error
in_err <- length(which(sign(agg_fit)!=Y_train))/length(Y_train)
out_err <- length(which(sign(agg_predict)!=Y_test))/length(Y_test) 


