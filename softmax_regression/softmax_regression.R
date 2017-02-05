#### Softmax regression ####
library(dplyr)

## Data
data(iris)

iris %>% group_by(Species) %>% summarize(MeanSepal.L = mean(Sepal.Length),
                                      MeanSepal.W = mean(Sepal.Width),
                                      MeanPetal.L = mean(Petal.Length),
                                      MeanPetal.W = mean(Petal.Width))


data_train  <- as.matrix(iris %>% mutate(intercept = 1,
               Sepal.L = (Sepal.Length-mean(Sepal.Length))/sd(Sepal.Length),
               Sepal.W = (Sepal.Width-mean(Sepal.Width))/sd(Sepal.Width),
               Petal.L = (Petal.Length-mean(Petal.Length))/sd(Petal.Length),
               Petal.W = (Petal.Width- mean(Petal.Width))/sd(Petal.Width))%>%
               select(-c(Species,Sepal.Length,Sepal.Width,Petal.Length,Petal.Width)))

y_train <- data.frame(setosa= iris$Species=="setosa",
                      versicolor= iris$Species=="versicolor",
                      virginica= iris$Species=="virginica")


## Plot
library(scatterplot3d)
library(RColorBrewer)

colors <- brewer.pal(3,name="Set2")
colors <- colors[as.numeric(iris$Species)]
scatterplot3d(iris[,1:3], pch = 16, color=colors,angle = 45)
legend("top", legend = levels(iris$Species),
       col =  brewer.pal(3,name="Set2"), pch = 16, 
       inset = -0.2, xpd = TRUE, horiz = TRUE)
##http://www.sthda.com/english/wiki/scatterplot3d-3d-graphics-r-software-and-data-visualization

## Function
# calculate the probabilities of belonging to each type
classprob <- function(x,w)
{
  signal<- x %*% t(w)
  signal <- exp(signal-apply(signal,1,function(row){return(max(row))}))
  return(sweep(signal,1,rowSums(signal),FUN= "/"))
}
# calculate the change of weight vectors
norm_vec <- function(x) sqrt(sum(x^2))
# calculate the gradient
compugrad <- function(x,y,w)
{
  prob <- classprob(x,w)
  for (j in 1:n) # to n-1 would give the same result since softmax is overparameterized
  {  
    gradient[j,] <- -colMeans(x*(y[,j]-prob[,j]))
  }
  return(gradient)
}
# calculate the cross-entropy error
compuerror <-function(x,y,w)
{
  prob <- classprob(x,w)
  E <- 0
  for ( j in 1:n)
  {
    E <- E-mean(y[,j]*log(prob[,j]))
  }
  return(E)
}

## Parameter
rate <- 0.1
batchs <- 10 # batch gradient descent
epsilon <- 10^-4
threshold <- 0.00001

batchl <- ceiling(nrow(data_train)/batchs)
n <- ncol(y_train)
f <- dim(data_train)[2]

## Initialization
w <- matrix(rep(0,f*n),ncol=f)
gradient <- matrix(rep(0,f*n),ncol=f)
Evec <- NULL
run <- 0
change <- 3

## Algorithm
while(change>threshold)
{  
  for (i in 1:batchl)
  {
    if (i < batchl)
    {
      y_train_b <- y_train[((i-1)*batchs+1):(i*batchs),]
      data_train_b <- data_train[((i-1)*batchs+1):(i*batchs),]
    }
    else
    {
      y_train_b <- y_train[((i-1)*batchs+1):dim(data_train)[1],]
      data_train_b <- data_train[((i-1)*batchs+1):dim(data_train)[1],]
    }
    gradient <- compugrad(data_train_b,y_train_b,w) 
    
    ## check gradient
    #check <- matrix(0,nrow=dim(gradient)[1],ncol=dim(gradient)[2])
    #for (j in 1:dim(gradient)[1])
    #{
    #  for (k in 1 :dim(gradient)[2])
    #  {
    #    w_plus <- w
    #    w_plus[j,k] <- w_plus[j,k]+epsilon
    #    E_plus <- compuerror(data_train_b,y_train_b,w_plus)  
    #    w_moins <- w
    #    w_moins[j,k] <- w_moins[j,k]-epsilon
    #    E_moins <- compuerror(data_train_b,y_train_b,w_moins)  
    #    check[j,k] <-abs(gradient[j,k]-(E_plus-E_moins)/(epsilon*2)) < epsilon
    #  }
    #}
    #if (mean(check)!=1)
    #{break}
    
    w <- w - rate*gradient
  }
  E <- compuerror(data_train,y_train,w)
  Evec <- rbind(Evec,E)
  if (run >1)
  {
    change <- abs(Evec[run,]-Evec[run-1,])
  }
  run <-run +1
}

## Result
plot(Evec,type="l",ylab="Error",xlab="Round") # Cross entropy error decreases

y_prob <- classprob(data_train,w)
listy <- unique(iris$Species)
y_hat <-  apply(y_prob,1, function(row){return(listy[which(row==max(row))])})
length(which(y_hat!=iris$Species)) # Classification error

table(Species=iris$Species,Predicted=y_hat)
