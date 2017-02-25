## Comapraison between unweighted PLA and weighted PLA over 200 rounds 
run <- 0
Eout_unw<- NULL
Eout_w <- NULL
Err_unw <- NULL
Err_w <- NULL
while(run<200)
{
#### Weighted PLA with noise
## Find a separating line
PX1 <-runif(2,min=-1,max=1)
PX2 <-runif(2,min=-1,max=1)
slope <- diff(PX2)/diff(PX2)
intercept <- PX2[1]-slope*PX1[1]

## Training points
ntrain <- 100
X1 <- runif(ntrain,min=-1,max=1)
X2 <- runif(ntrain,min=-1,max=1)
X <- cbind(rep(1,ntrain),X1,X2)
Y <- ifelse(X2>intercept+slope*X1,+1,-1)
## Add 5% errors
err <- sample(ntrain,ntrain*0.05)
Y[err] <- -Y[err] 

## Testing points
ntest <- 1000
X1_test <- runif(ntest,min=-1,max=1)
X2_test <- runif(ntest,min=-1,max=1)
X_test <- cbind(rep(1,ntest),X1_test,X2_test)
Y_test <- ifelse(X2_test>intercept+slope*X1_test,+1,-1)
## Add 5% errors
err <- sample(ntest,ntest*0.05)
Y_test[err] <- -Y_test[err] 

### Unweighted PLA
## Initialize
w<- c(0,0,0)
Y_hat <- X %*% w
count <- 1
change <- 1
maxit <- 1000
## PLA
Einvec <- NULL
while (count < maxit)
{
  miss_class <- which(sign(Y_hat)!=sign(Y))
  n <- ifelse(length(miss_class)>1,sample(miss_class,1),miss_class)
  w_temp <- w+(Y[n]%*% X[n,])
  Y_hat <- X %*% t(w_temp)
  Ein <- length(which(sign(Y_hat)!=Y))
  Einvec <- rbind(Einvec,Ein)
  if (count >1)
  {
    if (Einvec[count]<Einvec[count-1]){change <- 1}else{change <- 0}
  }
  if (change ==1)
  {
    w <- w_temp
  }
  Y_hat <- X %*% t(w)
  count <- count+1
}

Y_hat_test_uw <- X_test %*% t(w)

Eout_unw <- rbind(Eout_unw,length(which(sign(Y_hat_test_uw)!=Y_test))/length(Y_test)) 
Err_unw <- rbind(Err_unw,length(which(sign(Y_hat_test_uw)!=Y_test&Y_test==1)))  
### Weighted PLA
## Initialize
weight <- 9
w<- c(0,0,0)
Y_hat <- X %*% w
count <- 1
change <- 1
## PLA
Einvec <- NULL
while (count < 1000)
{
  miss_class <- which(sign(Y_hat)!=sign(Y))
  len1 <-length(which(Y[miss_class] ==1))
  lenm1 <-length(which(Y[miss_class] ==-1))
  prob <- ifelse(Y[miss_class] ==1,(len1*weight/(lenm1+len1*weight))/len1,(lenm1/(lenm1+len1*weight))/lenm1)
  n <- ifelse(length(miss_class)>1,sample(miss_class,1,prob=c(prob)),miss_class)
  w_temp <- w+(Y[n]%*% X[n,])
  Y_hat <- X %*% t(w_temp)
  Ein <- length(which(sign(Y_hat)!=Y&Y==1))*weight+length(which(sign(Y_hat)!=Y&Y==-1))
  Einvec <- rbind(Einvec,Ein)
  if (count >1)
  {
    if (Einvec[count]<Einvec[count-1]){change <- 1}else{change <- 0}
  }
  if (change ==1)
  {
    w <- w_temp
  }
  Y_hat <- X %*% t(w)
  count <- count+1
}
  
Y_hat_test_w <- X_test %*% t(w)
Eout_w <- rbind(Eout_w,length(which(sign(Y_hat_test_w)!=Y_test))/length(Y_test))
Err_w <- rbind(Err_w,length(which(sign(Y_hat_test_w)!=Y_test&Y_test==1)))
run <- run+1
}
  
## Out of sample error
mean(Eout_unw) 
mean(Eout_w)

## False negative rate
mean(Err_unw)
mean(Err_w)
