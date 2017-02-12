Logistic regression
================

In this document, we use logistic regression with stochastic gradient descent as optimization algorithm to classify different species of iris.

Preparing the data
------------------

The data used here to train toe model comes from the R default iris data set that includes the length and the width of the petals and the sepals of three different species of iris. Here, we will keep only two of the four features of iris- sepal length and sepal width.

``` r
library(dplyr)
```

    ## Warning: package 'dplyr' was built under R version 3.2.5

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
data(iris)
D <- iris %>% select(-c(Petal.Length,Petal.Width))
```

A quick summary table shows that the three species of iris indeed tend to differ in the length and the width of sepals.

``` r
D %>% group_by(Species) %>% summarize(meanSepal.L= mean(Sepal.Length),meanSepal.W= mean(Sepal.Width))
```

    ## # A tibble: 3 × 3
    ##      Species meanSepal.L meanSepal.W
    ##       <fctr>       <dbl>       <dbl>
    ## 1     setosa       5.006       3.428
    ## 2 versicolor       5.936       2.770
    ## 3  virginica       6.588       2.974

Before goinng on with logistic regression, we will first normalize the two features and add an intercept for each observation. Since we are doing binary classification, we are only going classify iris to Setosas and non-Setosas (Versicolors and Virginicas). We code the output (the species) of each observation to 1 for iris who are Setosa and -1 to those who are not.

``` r
X <- as.matrix(D  %>% mutate(Sepal.L.Norm =
                               Sepal.Length-mean(Sepal.Length),Sepal.W.Norm =
                               Sepal.Width-mean(Sepal.Width),intercept=1)%>%
                 select(-c(Species,Sepal.Length,Sepal.Width))) 
Y <- ifelse(D$Species=="setosa",1,-1)
```

Logistic regression
-------------------

Essentially, what logistic regression does is passing the signal, \(s\), which is the result of the dot product between a vector composed of the features of one observation \(X_{n}\) and a vector of weights w, through a logistic function, theta, defined as : \(theta(s) = e^s/(1+e^s)\). The output of the logistic function is bounded between 0 and 1.

The algorithm used to find the optimal weights vector is gradient descent, which consists of updating the weights vector to the opposite direction of the gradient of the cost function. Here, we use the cross-entropy error function as cost function. Unlike with perceptron and linearly separable data where the algorithm stops automatically when there is no more misclassified data, here, we need to define a stop condition for the algorithm. We will tell the algorithm to stop when the variation (Euclidean distance) between the weight vector updated in the present round and the weight vector from the last round is smaller than a threshold.

In the following codes, we define three functions to calculate the cross-entropy error, its gradient and the Euclidean distance between two weights vectors.

``` r
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
```

To start the algorithm, we initialize the weights vector to all zero. We also set the parameters to desired values. The two parameters we have to set are the rate of the gradient descent and the threshold of the variation of the weights vector below which the algorithm should stop.

``` r
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
```

Result
------

The following plot shows that the average cross-entropy error of the training data decreases over rounds.

``` r
plot(e.vec,type="l",xlab="Round",ylab="Average cross-entropy error")
```

![](logistic_regression_iris_files/figure-markdown_github/unnamed-chunk-6-1.png)

Using the final weights vector, we compute the signal for every observation and decides that a given observation is a Setosa if the logistic function on its signal results in a value that is greater than 0.5. In probabilistic terms, this means that the probability that the given observation is Setosa is greater than 0.5. By doing so, we successfully distinguished all the Setosas from the Non-Setosas in our training data. This allows us to say that our algorithm is working correctly.

``` r
Y.sig <- exp(X %*% t(w))/(1+exp(X%*%t(w)))
Y.predict <- ifelse(Y.sig >0.5,1,-1)
length(which(Y.predict != Y)) ## zero misclassified
```

    ## [1] 0