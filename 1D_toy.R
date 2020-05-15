library(mlt)
library(keras)
library(tensorflow)
library(tfprobability)
source('bern_utils.R')
source("model_utils.R")

NB = 10 #Number of Bernstein Polynomials

get_data_sin = function(){
  x = seq(0,5,length.out = 1500)
  y = 0.1*x+0.2+0.15*sin(x*2) + (0.01 + 0.015 * x)*rexp(500, 1) #For a exponential distribution
  return (list(x=x, y=y, dat=data.frame(x=x, y=y), scale=1))
}

get_data_sin_dist = function(x, y_in){
  offset = 0.1*x+0.2+0.15*sin(x*2)
  scale =  (0.01 + 0.015 * x)
  return(dexp((y_in - offset)/scale,1)/scale)
}

xy_dat = get_data_sin()
x = xy_dat$x
y = xy_dat$y
delta=abs(diff(range(y)))/4
my_min = min(y)-delta
my_max = max(y)+delta
plot(x,y, ylim=c(my_min, my_max))
xy_dat = get_data_sin()
max(xy_dat$y)
min(xy_dat$y)
x = xy_dat$x
y = xy_dat$y
dat=xy_dat$dat

delta=abs(diff(range(y)))/4
my_min = min(y)-delta
my_max = max(y)+delta

plot(x,y, ylim=c(my_min, my_max))

#### MLT ###############################################3
var_y <- numeric_var("y", support = c(my_min, my_max))
bb <- Bernstein_basis(var_y, order=NB, ui="increasing")
y_grid <- as.data.frame(mkgrid(bb, n = 500))
# set up model for mlt
ctm = ctm(bb, shift=~x, data=dat, todistr="Normal") 
mlt_fit <- mlt(ctm, data = dat, verbose=TRUE)
(logLik_mlt = logLik(mlt_fit)/ nrow(dat))
mlt_fit$coef

preds = predict(mlt_fit, newdata=dat,  q=y_grid$y, type='density')
plot(x,y, ylim=c(my_min, my_max), col='grey')
strech=0.25
n=length(y)  # number of simulated points
for (idx in c(n*0.1,n*0.25,n*0.5,n*0.75,n*0.9)){
  f = preds[,idx]
  lines(strech*f+x[idx],y_grid$y)
  abline(v=x[idx], lty=2)
}


# MLT Network Model 
xx = tf$Variable(as.matrix(x, ncol=1), dtype='float32')
yy = tf$Variable(as.matrix(y, ncol=1), dtype='float32')

len_theta = as.integer(NB + 1L)
T_OUT = 100
run = 1
history = make_hist()

source('model_7.R')
# Sometimes Error in on_load(), ignore 
model_7 = new_model_7(len_theta = len_theta, x_dim = 1, y_range=1)
model_7$name = 'model_7'
history = model_train(model_7, history, xx, yy,xx, yy, T_STEPS = 1500)


NLLS = 0
#--------- Creating the plot
plot(x,y,col='grey', xlim=c(0,5.5), ylim=c(-0.2,0.8))
streches = c(0.1, 0.1, 0.1)
cc = 0
for (i in c(n*0.01,n*0.25,n*0.5)){
  cc = cc + 1
  strech = 0.1#streches[cc]
  NLL = model_test(model_7,xx[i,,drop=FALSE],yy[i,,drop=FALSE])
  print(NLL)
  NLLS = NLLS + NLL
  int_steps = 500 #Steps for the integration
  
  y_start = -0.2
  y_end = 1.0
  ret = model_get_p_y(model_7, xx[i,,drop=FALSE], y_start, y_end, int_steps)
  print(paste0(i, '  ',round(sum(ret$p_y)/(int_steps/(y_end - y_start)),3)))
  f = ret$p_y
  lines(strech*f+x[i],ret$y,col='black')
  abline(v=x[i], lty=2)
  f = preds[,i]
  lines(strech*f+x[i],y_grid$y, lty=2, col='black')

  if (TRUE){
    f = get_data_sin_dist(x[i], ret$y)
    lines(strech*f+x[i],ret$y, lty=1, col='blue')
  }
}
NLLS / 100


