
## Generate random points x1,x2 with intercept
X1 <- runif(100,min=-1,max=1)
X2 <- runif(100,min=-1,max=1)
X0 <- rep(1,100)
D <- cbind(X0,X1,X2)

## Find a separating line
PX1 <-runif(2,min=-1,max=1)
PX2 <-runif(2,min=-1,max=1)
slope <- diff(PX2)/diff(PX2)
intercept <- PX2[1]-slope*PX1[1]

## Y = 1 for points above the separating line, -1 for those below
Y <- ifelse(X2>intercept+slope*X1,+1,-1)
D.plot <- as.data.frame(D) # dataframe for graphs later

## Initialize
wdf <- NULL # df to collect weights in each round for graph
w <- c(0,0,0)
Y_hat <- D %*% w
count <- 0
wdf <- rbind(wdf,w)
## Perceptron learning algorithm
while (any(sign(Y_hat)!=sign(Y)))
{
  miss_class <- which(sign(Y_hat)!=sign(Y))
  ifelse(length(miss_class)>1,n <- sample(miss_class,1),n <- miss_class)
  w <- w+(Y[n]%*% D[n,])
  wdf <- rbind(wdf,w)
  Y_hat <- D %*% t(w)
  count <- count+1
}

## Draw plot
drawplot <- function(n)
{
  int_w <- -wdf[n,1]/wdf[n,3]
  slope_w <- -wdf[n,2]/wdf[n,3]
  a <- ggplot(D.plot,aes(D.plot$X1,D.plot$X2))+
    geom_point(aes(color=as.factor(Y)),show.legend = F)+
    geom_abline(intercept = intercept,slope=slope)+
    geom_abline(intercept = int_w,slope=slope_w,linetype="dashed")+
    labs(x="X",y="Y",color="")
  print(a)
}

drawplot(67)
