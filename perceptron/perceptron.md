Perceptron Learning Algorithm
================

This document demonstrates the application of the Perceptron Learning Algorithm (PLA) to some linearly separable 2D data for the purpose of classification.

Preparing the data
------------------

First, we generate 100 random points with coordinates X1, X2 and an intercept X0:

``` r
X1 <- runif(100,min=-1,max=1)
X2 <- runif(100,min=-1,max=1)
X0 <- rep(1,100)
X <- cbind(X0,X1,X2)
```

We then find a random line on he plane to separate these points into two classes. Points above the line are assigned +1 whereas points below are assigned -1.

``` r
PX1 <-runif(2,min=-1,max=1)
PX2 <-runif(2,min=-1,max=1)
slope <- diff(PX2)/diff(PX2)
intercept <- PX2[1]-slope*PX1[1]
Y <- ifelse(X2>intercept+slope*X1,+1,-1)
X.plot <- as.data.frame(X) # dataframe for plot use
```

The points and the seperating line look like the following :

    ## Warning: package 'ggplot2' was built under R version 3.2.5

``` r
ggplot(X.plot,aes(X.plot$X1,X.plot$X2))+
    geom_point(aes(color=as.factor(Y)),show.legend = F)+
    geom_abline(intercept = intercept,slope=slope)+
    labs(x="X1",y="X2",color="")
```

![](perceptron_files/figure-markdown_github/unnamed-chunk-4-1.png)

PLA
---

The goal of the PLA here is to find a line that can correctly classify the points. This, of course, is done without the knowledge of the true seperating that we just created. What the PLA does essentially is to pick a misclassified point and