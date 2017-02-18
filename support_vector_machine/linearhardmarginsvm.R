library(quadprog)
Result <- NULL
Eout <- NULL
run <- 0
while(run < 200)
{
  repeat{
  ## Pick a line
  x1l <- runif(2,-1,1)
  x2l <- runif(2,-1,1)
  slopel <- diff(x2l )/diff(x1l)
  interceptl <- x2l[1]-slopel*x1l[1]
  ## Create training data
  n <- 50
  x1_tr <- runif(n,-1,1)
  x2_tr <- runif(n,-1,1)
  intercept_tr <- rep(1,n)
  data_tr<- cbind(intercept_tr,x1_tr,x2_tr)
  y_tr <- ifelse(x2_tr>interceptl+slopel*x1_tr,+1,-1)
  if(any(y_tr==1)&any(y_tr==-1))
  {break}
  }
  ## Create test data
  n_ts <- 10000
  x1_ts <- runif(n_ts,-1,1)
  x2_ts <- runif(n_ts,-1,1)
  intercept_ts <- rep(1,n_ts)
  data_ts <- cbind(intercept_ts,x1_ts,x2_ts)
  y_ts <- ifelse(x2_ts>interceptl+slopel*x1_ts,+1,-1)
  
  ## PLA training
  data_pla <- data_tr
  w_pla <- c(0,0,0)
  y_pla <- w_pla %*% t(data_pla)
  
  while (any(sign(y_pla)!=sign(y_tr)))
  {
    l <- which(sign(y_pla)!=sign(y_tr))
    ifelse(length(l)>1,choice <- sample(l,1),choice <- l)
    w_pla <- w_pla+(y_tr[choice]%*% data_pla[choice,])
    y_pla <- w_pla%*% t(data_pla)
  }
  ## PLA test
  pla_ts <- sign(w_pla%*% t(data_ts))
  PLA <-length(which(pla_ts!=y_ts))/n_ts 
  
  ## SVM training
  data_svm <- data_tr[,-1]
  M <- matrix(NA,nrow = n,ncol=n)
  for (i in 1:n)
  {for (j in 1:n)
  {
    M[i,j] <- y_tr[i]*y_tr[j]*(t(data_svm[i,])%*%data_svm[j,])
  }
  }
  d <- rep(1,n)
  A <- cbind(y_tr,diag(n))
  b <- rep(0,n+1)
  modification <- 1e-13
  M <- M + modification * diag(n)
  QP<-solve.QP(Dmat = M,  dvec = d,Amat=A,bvec=b,meq=1)
  alpha <- c(QP$solution)
  w_svm <- (QP$solution * y_tr)%*% data_svm
  sv <- which(QP$solution==max(QP$solution)) #pick the largest alpha to solve for b
  b <- y_tr[sv]-w_svm%*%data_svm[sv,]
  w_svm_f <- cbind(b,w_svm)
  
  # SVMEout
  svm_ts<- sign(w_svm_f %*% t(data_ts))
  SVM <-length(which(svm_ts!=y_ts))/n_ts
  Eout <- rbind(Eout,cbind(SVM,PLA))
  run <- run +1
}
Eout <- data.frame(Eout)
length(which(Eout$SVM<Eout$PLA))/nrow(Eout)

library(ggplot2)
library(reshape2)
Eout <- melt(Eout)
ggplot(Eout, aes(value, fill = variable)) + 
  geom_histogram(alpha = 0.5, position = 'identity')+
  xlab("Out-of-sample error")


