#### Softmax regression ####
## Data
data(iris)
listy <- as.character(unique(iris$Species))
data.train  <- as.matrix(iris[,-5])
y.train <- as.character(iris[,5])

## Function
hcal <- function(data,x){
  signal<- data %*% t(x)
  signal <- exp(signal-apply(signal,1,function(row){return(max(row))}))
  return(sweep(signal,1,rowSums(signal),FUN= "/"))
}
norm_vec <- function(x) sqrt(sum(x^2))
## Parameter
rate <- 0.1
n <- length(listy)
f <- dim(data.train)[2]
batchs <- 10
batchl <- ceiling(nrow(data.train)/batchs)
epsilon <- 10^-4
threshold <- 0.00005
## Initialization
w <- matrix(rep(0,f*n),ncol=f)
gradiant <- matrix(rep(0,f*n),ncol=f)
Evec <- NULL
run <- 0
change <- 3
E <- 100

while(change>threshold)
{  
  for (i in 1:batchl)
  {
  if (i < batchl)
  {
    y.train.b <- y.train[((i-1)*batchs+1):(i*batchs)]
    data.train.b <- data.train[((i-1)*batchs+1):(i*batchs),]
    h.b <- hcal(data.train.b,w)
  }
  else
  {
    y.train.b <- y.train[((i-1)*batchs+1):dim(data.train)[1]]
    data.train.b <- data.train[((i-1)*batchs+1):dim(data.train)[1],]
    h.b <- hcal(data.train.b,w)
  }
  for (j in 1:(n-1))
  {  
    y.temp <- ifelse(y.train.b ==listy[j], 1,0)
    gradiant[j,] <- -colSums(data.train.b*(y.temp-h.b[,j]))/nrow(data.train.b)
  }
  w <- w - rate*gradiant
  }
  h <- hcal(data.train,w)
  E <- 0
  for ( j in 1:n)
  {
    y.temp <- ifelse(y.train ==listy[j], 1,0)
    E <- E-sum(y.temp*log(h[,j]))/nrow(data.train)
  }
  Evec <- rbind(Evec,E)
  if (run >1)
  {
    change <- abs(Evec[run,]-Evec[run-1,])
  }
  run <-run +1
}

plot(Evec,type="l") # Cross entropy error decreases

signal<- data.train %*% t(w)
signal <- exp(signal-apply(signal,1,function(row){return(max(row))}))
h <- sweep(signal,1,rowSums(signal),FUN= "/")
y_hat <- apply(h,1, function(row){return(listy[which(row==max(row))])})
length(which(y_hat==y.train))/nrow(data.train) # Classification error
