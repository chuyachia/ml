library(dplyr)
## Data
data <- read.table("glass.data",dec=".",sep=",")
X_1 <- data %>% select(-c(V1,V2,V11)) %>% apply(2,FUN=function(column){return((column-mean(column))/sd(column))} )
n_data <- nrow(X_1)
n_input <- ncol(X_1)
Y_type <- unique(data$V11)


## Parameter
# binary
n_output <- 1
target <- Y_type[3]
Y <- ifelse(data$V11==target,1,0)
# multi
n_output <- length(Y_type)
data$V11 <- as.factor(data$V11)
Y <- model.matrix(~data$V11-1,data=data)
# model
rate <-0.06
n_hidden <- 20

## Initialize
W_1 <- matrix(rnorm(n_input*n_hidden,0,0.01),nrow=n_input,ncol=n_hidden)
b_1 <- matrix(0,nrow=1,ncol=n_hidden)

W_2 <- matrix(rnorm(n_hidden*n_output,0,0.01),nrow=n_hidden,ncol=n_output)
b_2 <- matrix(0,nrow=1,ncol=n_output)

run <- 0
evec <- NULL

while(run < 35000)
{  
signal_1 <- sweep(X_1 %*% W_1, 2, b_1,"+")
X_2 <- pmax(signal_1,0)
  
signal_2 <- sweep(X_2 %*% W_2, 2, b_2,"+")
signal_2 <- exp(signal_2)
X_3 <- sweep(signal_2,1,rowSums(signal_2),"/")

e <- -colSums(log(X_3) * Y)
e <- sum(e)
evec <- rbind(evec,e)

delta_2 <- X_3-Y
delta_2 <- delta_2/n_data
delta_1 <- delta_2 %*% t(W_2)
delta_1[X_2 <= 0] <- 0

gradient_W2 <- t(X_2) %*% delta_2
gradient_b2 <- colSums(delta_2)

gradient_W1 <- t(X_1) %*% delta_1
gradient_b1 <- colSums(delta_1)

#gradient_W1 <- t(X_1) %*%(X_2-Y)
#gradient_W1 <- gradient_W1 / n_data
#gradient_b1 <- sum(X_2-Y)/n_data

W_2 <- W_2 - rate*gradient_W2
b_2 <- b_2- rate*gradient_b2

W_1 <- W_1 - rate*gradient_W1
b_1 <- b_1- rate*gradient_b1

run <- run +1
}
plot(evec,type="l")

signal_2 <- sweep(X_2 %*% W_2, 2, b_2,"+")
signal_2 <- exp(signal_2)
X_3 <- sweep(signal_2,1,rowSums(signal_2),"/")
y.predict <- apply(X_3, 1, function(row){Y_type[which.max(row)]})
length(which(y.predict==data$V11))/n_data

