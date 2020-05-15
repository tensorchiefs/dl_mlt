# Model as described in https://arxiv.org/abs/2004.00464 Eq (8)
# \alpha(x) \left( \sum_{i=0}^M {\tt Be_i}(\sigma(a(x) \cdot y - b(x)) \frac{\vartheta_i(x)}{M+1} \right) - \beta(x)

# model_g = a (paper)
# model_s = b (paper)
# model =  theta (paper)
# model_a = alpha (paper)
# model_beta = beta (paper)

# Definition of the network model
.make_model = function(len_theta, x_dim){ 
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units=(10), input_shape = c(x_dim), activation = 'tanh') %>% 
    layer_dense(units=(100), activation = 'tanh') %>% 
    layer_dense(units=len_theta) %>% 
    layer_activation('linear') 
  return (model)
}

# Private definition of the network model
.make_model_g = function(x_dim){ 
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units=(10), input_shape = c(x_dim), activation = 'tanh') %>% 
    layer_dense(units=(100), activation = 'tanh') %>% 
    layer_dense(units=1) %>% 
    layer_activation('softplus') #Needs to be positive
  return (model)
}

# Private definition of the network model
.make_model_s = function(x_dim){ 
  modeld <- keras_model_sequential() 
  modeld %>% 
    layer_dense(units=(10), input_shape = c(x_dim), activation = 'tanh') %>% 
    layer_dense(units=(100), activation = 'tanh') %>% 
    layer_dense(units=1) %>% 
    layer_activation('linear') 
  return (modeld)
}


###############################
# Linear scale term gives coefficients for the predictor (beta)
.make_model_beta = function(x_dim){
  model_beta <- keras_model_sequential() 
  model_beta %>% 
    layer_dense(10, activation='tanh', input_shape = x_dim) %>% 
    layer_dense(50, activation='tanh') %>% 
    layer_dense(10, activation='tanh') %>% 
    layer_dense(5, activation='tanh') %>% 
    layer_dense(1, activation='linear')
  return(model_beta)
}

###############################
# Linear shift term gives coefficients for the predictor (beta)
.make_model_a = function(x_dim){
  model_a <- keras_model_sequential() 
  model_a %>% 
    layer_dense(10, activation='tanh', input_shape = x_dim) %>% 
    layer_dense(50, activation='tanh') %>% 
    layer_dense(5, activation='tanh') %>% 
    layer_dense(1, activation='softplus') #We need to be positive
  return(model_a)
}


#"Private" and complete constructor. This constructor verifies the input.
new_model_7 = function(len_theta = integer(), x_dim, y_range, bs = -1,eta_term=TRUE, reg_factor=-1, a_term=TRUE){
  stopifnot(is.integer(len_theta))
  stopifnot(len_theta > 0)
  model_beta = NULL
  if (eta_term){
    model_beta = .make_model_beta(x_dim)
  }
  model_a = NULL
  if (a_term){
    model_a = .make_model_a(x_dim)
  }
  
  structure( #strcutur is a bunch of data with 
    list( #bunch of data
      len_theta=len_theta,  
      x_dim = x_dim,
      optimizer = tf$keras$optimizers$Adam(learning_rate=0.0001),
      model = .make_model(len_theta, x_dim),
      model_g = .make_model_g(x_dim),
      model_s = .make_model_s(x_dim),
      model_beta = model_beta,
      model_a = model_a,
      bernp = make_bernp(len_theta),
      reg_factor = reg_factor,
      y_range = y_range,
      bs = bs,
      name = 'model_7'
    ),
    class = "model_7" #class attribute set so it's a class
  )  
}

add_squarred_weights_penalty = function(weights, NLL, lambda=0.05){
  for (w in weights){
    NLL = NLL + lambda*tf$reduce_sum(tf$math$square(w)) #L2
    #NLL = NLL + lambda*tf$reduce_sum(tf$math$abs(w)) #L1
  }
  return(NLL)
}

calc_NLL = function(out_hy, g, s, y_train, y_range, out_eta, bernp, a_x) {
  y_tilde = g*y_train - s #TODO make nicer
  y_tilde_2 = tf$math$sigmoid(y_tilde)
  # print(summary(as.numeric(y_tilde)))
  # print('g')
  # print(summary(as.numeric(y_tilde_2)))
  theta_im = to_theta(out_hy)
  if (!is.null(out_eta)){
    z = a_x[,1] * eval_h(theta_im, y_i = y_tilde_2, beta_dist_h = bernp$beta_dist_h) - out_eta[,1]
  } else {
    z = a_x[,1] * eval_h(theta_im, y_i = y_tilde_2, beta_dist_h = bernp$beta_dist_h) 
  }
  h_y_dash_part1 = a_x[,1] * eval_h_dash(theta_im, y_tilde_2, beta_dist_h_dash = bernp$beta_dist_h_dash) 
  h_y_dash_part2 = g 
  #sig' = sig(1-sig)
  l_h_y_dash_part3 = tf$math$log(tf$math$sigmoid(y_tilde)) +  tf$math$log((1 - tf$math$sigmoid(y_tilde)))
  return(-tf$math$reduce_mean(
    bernp$stdnorm$log_prob(z) + 
    tf$math$log(h_y_dash_part1) + 
    tf$math$log(h_y_dash_part2) +
    l_h_y_dash_part3) + 
    log(y_range)
    )
}

train_step = function(x_train, y_train, model){
  with(tf$GradientTape() %as% tape, {
    out_hy = model$model(x_train)
    g = model$model_g(x_train)
    sss = model$model_s(x_train)
    model_beta = model$model_beta
    if (!is.null(model_beta)){
      beta_x = model$model_beta(x_train)  
    } else {
      beta_x = tf$zeros_like(y_train)
    }
    model_a = model$model_a
    if (!is.null(model_a)){
      a_x = model$model_a(x_train)  
    } else {
      a_x = tf$ones_like(y_train)
    }
    NLL = calc_NLL(out_hy, g, sss, y_train, model$y_range, out_eta = beta_x, bernp=model$bernp, a_x=a_x)
    if (model$reg_factor > 0){
      NLL = add_squarred_weights_penalty(weights=model$model$trainable_variables, NLL=NLL, lambda=model$reg_factor)
      NLL = add_squarred_weights_penalty(weights=model$model_g$trainable_variables, NLL=NLL, lambda=model$reg_factor)
      NLL = add_squarred_weights_penalty(weights=model$model_s$trainable_variables, NLL=NLL, lambda=model$reg_factor)
      if (!is.null(model_a)) NLL = add_squarred_weights_penalty(weights=model$model_a$trainable_variables, NLL=NLL, lambda=model$reg_factor)
      if (!is.null(model_beta)) NLL = add_squarred_weights_penalty(weights=model$model_beta$trainable_variables, NLL=NLL, lambda=model$reg_factor)
    }
  })
  
  #Creating a list for all gradients
  n = 3
  tvars = list(model$model$trainable_variables, 
               model$model_g$trainable_variables,
               model$model_s$trainable_variables) 
  if (!is.null(model_a)) {
    n = n + 1
    tvars[[n]] =  model$model_a$trainable_variables
  }
  if (!is.null(model_beta)) {
    n = n + 1
    tvars[[n]] = model$model_beta$trainable_variables
  }
  
  #Calculation of the gradients
  grads = tape$gradient(NLL, tvars)
  for (i in 1:n){
    model$optimizer$apply_gradients(
      purrr::transpose(list(grads[[i]], tvars[[i]]))
    )  
  }
  return(NLL)
}


#train_step_au = train_step#tf_function(train_step) 
train_step_au = tf_function(train_step) #Using tf_function to speed up calculation
model_train = function(model, history, x_train, y_train, x_test, y_test,save_model = FALSE, T_STEPS){
  start_time = Sys.time()
  bs = model$bs
  print(paste0('Training model 7 traing ',x_train$shape, ' testing ', x_test$shape, ' ',bs))
  for (r in 1:T_STEPS){
    #idx_batch = sample(start_index:(length(y_train)-1+start_index), bs)
    if (bs > 0){
      idx_batch = sample(1:(length(y_train)-1), bs) #To be on the safe side
      x_train_batch = tf$Variable(x_train$numpy()[idx_batch,], dtype='float32')#does not work x_train[idx_batch]
      x_train_batch = tf$reshape(x_train_batch, shape=c(bs,-1L))
      dd = tf$Variable(y_train$numpy()[idx_batch],dtype='float32')
      y_train_batch = tf$reshape(dd, shape=c(bs,1L))#y_train[idx_batch]
      #y_train_batch = y_train_batch$reshape(-1L,1L)  
    } else {
      x_train_batch = x_train
      y_train_batch = y_train
    }
    l  = train_step_au(x_train=x_train_batch, y_train=y_train_batch, model=model)  
    if (r %% T_OUT == 0){
      out_hy = model$model(x_test)
      g = model$model_g(x_test)
      s = model$model_s(x_test)
      model_beta = model$model_beta
      if (!is.null(model_beta)){
        beta_x = model$model_beta(x_test)  
      } else {
        beta_x = tf$zeros_like(y_test)
      }
      model_a = model$model_a
      if (!is.null(model_a)){
        a_x = model$model_a(x_test)  
      } else {
        a_x = tf$ones_like(y_test)
      }
      NLL = calc_NLL(out_hy, g, s, y_test, model$y_range, out_eta = beta_x, bernp=model$bernp, a_x=a_x)
      print(paste(r, 'likelihood (in optimize) ' ,l$numpy(), 'likelihood (in test) ',NLL$numpy()))
      history = rbind(history, c(r, run, l$numpy() , NLL$numpy(), model$name))
    } 
  }
  end_time = Sys.time() 
  if (save_model){
    save_model_hdf5(model, paste0('boston_cv_model7_F',run,'_Steps_',T_STEPS))
  }
  return(history)
}

# Application of the trained model on the test set returning the likelihood
model_test = function(model, x_test, y_test){
  out_hy = model$model(x_test)
  g = model$model_g(x_test)
  s = model$model_s(x_test)
  if (!is.null(model$model_beta)){
    beta_x = model$model_beta(x_test)  
  } else {
    beta_x = tf$zeros_like(y_test)
  }
  model_a = model$model_a
  if (!is.null(model_a)){
    a_x = model$model_a(x_test)  
  } else {
    a_x = tf$ones_like(y_test)
  }
  NLL = calc_NLL(out_hy, g, s, y_test, model$y_range, out_eta = beta_x, bernp=model$bernp, a_x=a_x)
  #NLL = calc_NLL(out_hy, g, s, y_test, model$y_range,  out_eta = model$beta_x, bernp=model$bernp)
  return(NLL$numpy())
}

model_get_p_y = function(model, x, from, to, length.out){
  bernp = model$bernp
  #stopifnot(x$shape[start_index] == 1) #We need a single row
  y_cont = keras_array(matrix(seq(from,to,length.out = length.out), nrow=length.out,ncol=1))
  
  out_hy = model$model(x)
  theta_im = to_theta(out_hy)
  
  g = model$model_g(x)
  s = model$model_s(x)
  
  model_a = model$model_a
  if (!is.null(model_a)){
    a_x = model$model_a(x)  
  } else {
    a_x = tf$ones_like(y_cont)
  }
  
  
  y_tilde = g*y_cont - s
  y_tilde_2 = tf$math$sigmoid(y_tilde)
  theta_rep = k_tile(theta_im, c(length.out, 1))
  #print(theta_rep)
  if(is.null( model$model_beta)){
    z =  a_x[,1] * eval_h(theta_rep, y_tilde_2, beta_dist_h = bernp$beta_dist_h)
  } else{
    beta_x = model$model_beta(x)
    z =  a_x[,1] * eval_h(theta_rep, y_tilde_2, beta_dist_h = bernp$beta_dist_h) - beta_x[,1]
  }
  
  p_y = bernp$stdnorm$prob(z) * as.array(a_x[,1] * eval_h_dash(theta_rep, y_tilde_2, beta_dist_h_dash = bernp$beta_dist_h_dash))
  p_y = tf$transpose(p_y * g) * tf$math$sigmoid(y_tilde) * (1.0 - tf$math$sigmoid(y_tilde))
  
  df = data.frame(
    y = seq(from,to,length.out = length.out),
    p_y = as.numeric(p_y),
    h = z$numpy()
    )
  df$y_tilde = as.numeric(y_tilde)
  df$y_tilde_2 = as.numeric(y_tilde_2)
  return (df)
}








