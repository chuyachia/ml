#### Perceptron learning algorithm ####
## Generate random points x1,x2 with intercept
countvec <- NULL
Eoutvec <- NULL
while (length(countvec)<100) # repeat 100 runs
{
  X1 <- runif(100,min=-1,max=1)
  X2 <- runif(100,min=-1,max=1)
  X0 <- rep(1,100)
  X <- cbind(X0,X1,X2)

  ## Find a separating line
  PX1 <-runif(2,min=-1,max=1)
  PX2 <-runif(2,min=-1,max=1)
  slope <- diff(PX2)/diff(PX2)
  intercept <- PX2[1]-slope*PX1[1]

  ## Y = 1 for points above the separating line, -1 for those below
  Y <- ifelse(X2>intercept+slope*X1,+1,-1)
  #X.plot <- as.data.frame(X) # dataframe for plot use

  ## Initialize
  w<- c(0,0,0)
  Y_hat <- X %*% w
  count <- 0
  ## Perceptron learning algorithm
  while (any(sign(Y_hat)!=sign(Y)))
  {
    miss_class <- which(sign(Y_hat)!=sign(Y))
    ifelse(length(miss_class)>1,n <- sample(miss_class,1),n <- miss_class)
    w <- w+(Y[n]%*% X[n,])
    Y_hat <- X %*% t(w)
    count <- count+1
  }
  countvec <- rbind(countvec,count)
  
  ## Testing out-of-sample error with 1000 
  X1_test <- runif(1000,min=-1,max=1)
  X2_test <- runif(1000,min=-1,max=1)
  X0_test <- rep(1,1000)
  X_test <- cbind(X0_test,X1_test,X2_test)
  Y_test <- ifelse(X2_test>intercept+slope*X1_test,+1,-1)

  Y_hat_test <- X_test %*% t(w)
  Eout <- length(which(sign(Y_hat_test)!=Y_test))/length(Y_test) # classification error
  Eoutvec <- rbind(Eoutvec,Eout) 
}
mean(Eoutvec) ## Average classification error of 100 rounds
table(predicted=sign(Y_hat_test),observed=Y_test)
