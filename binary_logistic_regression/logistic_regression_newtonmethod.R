### Logistic regression with Newton's method

## Data
library(dplyr)
data(iris)
x <- iris %>% select(-Species) %>% mutate(intercept = 1)
x <- as.matrix(x)
y <- ifelse(iris$Species=="virginica",1,0)

## Function
# probability
prob <- function(x,w){
  exp(x %*%w)/(1+exp(x%*%w))}


# derivative of likelihood
# first order
fprime1 <- function(x,w,p){
  t(x) %*% (y-p)
}
# second order
fprime2 <- function(x,w,p){
  M <- diag(n)
  diag(M) <- sapply(c(1:n),function(x){p[x]*(1-p[x])}) 
  return(-t(x) %*% M %*% x)
}
# distance between two weight vectors
norm_vec <- function(x) sqrt(sum(x^2))

## Parameters
n <- nrow(x)
wvec <- NULL
run <- 1
w <- rep(0,5)
thres <- 0.001
change <- 100

## Algorithm
while(change > thres)
{
p <- prob(x,w)
w <-w+solve(-fprime2(x,w,p)) %*% fprime1(x,w,p)
wvec <- rbind(wvec,t(w))
if (run > 1)
{
  change <- norm_vec(wvec[run,]-wvec[run-1,])
}
run <- run+1
}

## Result
p_final <- prob(x,w)
y_pred <- ifelse(p_final>0.5,1,0)
length(which(y_pred == y))/n # accuracy

-solve(fprime2(x,w,p)) ##variance covariance matrix

## Using R package
date_package <- iris %>% mutate(intercept = 1,Species=ifelse(Species=="virginica",1,0))
result_package <- glm(Species~.,data=date_package, family = "binomial")
y_pred <- ifelse(predict(result_package)>0.5,1,0)
length(which(y_pred == y))/n # accuracy
vcov(result_package) ##variance covariance matrix
