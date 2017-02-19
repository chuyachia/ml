library(dplyr)
library(ggplot2)
data(iris)
D <- iris %>% select(-c(Petal.Length,Petal.Width))

X <- as.matrix(D  %>% mutate(Sepal.L.Norm = Sepal.Length-mean(Sepal.Length),
                             Sepal.W.Norm=Sepal.Width-mean(Sepal.Width),intercept=1)%>%
                 select(-c(Species,Sepal.Length,Sepal.Width))) 
Y <- ifelse(D$Species=="setosa",1,-1)

## Function
# Cross-entropy error
E <- function(x,y,z) 
{
  log(1+exp(-y%*%z%*%x))
}
# Gradient 
grad <- function(x,y,z)
{
  (-y%*%x)/c(exp(y%*%z%*%x)+1)
}
# Function to calculate Euclidean distance
norm_vec <- function(x) sqrt(sum(x^2))

## Parameter
rate <- 0.01
thres <- 0.01
## Initialization
w <- rep(0,ncol(X))
w.vec <- NULL
w.vec <- rbind(w.vec,w)
e <- NULL
e.vec <- NULL
epoch <- 1
w.var <-1 

while (w.var>=thres)
{
  for (i in 1 : nrow(X))
  {
    e[i] <- E(X[i,],Y[i],w.vec[epoch,])
  }
  e.vec <- rbind(e.vec,mean(e))
  
  seq <-sample(c(1:nrow(X)),nrow(X),replace=F)
  for (i in 1:100) 
  {
    gradient <- grad(X[seq[i],],Y[seq[i]],w)
    w <- w -rate%*%gradient
  }
  w.vec <- rbind(w.vec,w)
  epoch = epoch+1
  w.var <-norm_vec(w.vec[epoch-1,]-w.vec[epoch,])
  
}

Y.sig <- exp(X %*% t(w))/(1+exp(X%*%t(w)))
Y.predict <- ifelse(Y.sig >0.5,1,-1)
length(which(Y.predict != Y)) ## zero misclassified observation