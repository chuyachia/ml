#### logistic regression ####

## Cross-entropy error
E <- function(x,y,z) 
{
  log(1+exp(-y%*%z%*%x))
}
## Gradient 
grad <- function(x,y,z)
{
  (-y%*%x)/c(exp(y%*%z%*%x)+1)
}
## Weights vector variation
norm_vec <- function(x) sqrt(sum(x^2))


## Repeat 100 time the algorithm
run <- 1
epoch.vec <- NULL
Eout.vec <- NULL
Eout_class.vec <- NULL
while(run<100)
{
  ## Find boundary line
  x1.l <- runif(2,-1,1)
  x2.l <- runif(2,-1,1)
  slope.l <- diff(x2.l )/diff(x1.l)
  intercept.l <- x2.l[1]-slope.l*x1.l[1]
  ## Create 100 random points and evaluate y
  x1.p <- runif(100,-1,1)
  x2.p <- runif(100,-1,1)
  intercept.p <- rep(1,100)
  data <- cbind(intercept.p,x1.p,x2.p)
  y.p <- ifelse(x2.p>intercept.l+slope.l*x1.p,+1,-1)
  
  
  ## Initialize weights, epoch
  ## Create a vector to collect all final weight until convergence
  w <- c(0,0,0)
  w.vec <- NULL
  w.vec <- rbind(w.vec,w)
  rate <- 0.01
  
  epoch <- 1
  w.var <-1 # Initialize to some values greater than threshold 0.01
  thres <- 0.01 
  while (w.var>=thres) #convergence criteria
  {
    seq <-sample(c(1:100),100,replace=F)
    #gradient descent for random permuation of N=100
    for (i in 1:100) 
    {
      gradient <- grad(data[seq[i],],y.p[seq[i]],w)
      w <- w -rate%*%gradient
    }
    w.vec <- rbind(w.vec,w)
    epoch = epoch+1
    w.var <-norm_vec(w.vec[epoch-1,]-w.vec[epoch,])
  }
  #out-of-samle errors using 1000 test points
  x1.out <- runif(1000,-1,1)
  x2.out <- runif(1000,-1,1)
  intercept.out <- rep(1,1000)
  data.out <- cbind(intercept.out,x1.out,x2.out)
  y.out <- ifelse(x2.out>intercept.l+slope.l*x1.out,+1,-1)
  y.out.sig <- exp(data.out %*% t(w))/(1+exp(data.out%*%t(w))) 
  y.out.hat <- ifelse(y.out.sig>0.5,1,-1)
  Eout_class <- length(which(y.out.hat!= y.out))/length(data.out)
  Eout_class.vec <-rbind(Eout_class.vec,Eout_class)
  Eout <- NULL
  for (i in 1:1000)
  {Eout[i] <- E(data.out[i,],y.out[i],w.vec[epoch,])}
  Eout.vec<- cbind(mean(Eout),Eout.vec)
  epoch.vec <- cbind(epoch,epoch.vec)
  run = run+1
}

mean(Eout.vec) # Average cross entropy error of 100 rounds
mean(Eout_class.vec) # Average classification error of 100 rounds
mean(epoch.vec) # Average number of runs needed