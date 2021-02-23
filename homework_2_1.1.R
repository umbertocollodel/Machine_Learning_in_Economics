# Installs packages if not already installed, then loads packages -----
list.of.packages <- c("SuperLearner", "ggplot2", "glmnet", "clusterGeneration", "mvtnorm", "xgboost",
                      "crayon","tidyverse")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))


# Custom function to generate the data -----

generate_data <- function(N=500, k=50, true_beta=1) {
  # DGP inspired by https://www.r-bloggers.com/cross-fitting-double-machine-learning-estimator/ 
  # Generates a list of 3 elements, y, x and w, where y and x are N 
  # times 1 vectors, and w is an N times k matrix. 
  #
  # Args:
  #   N: Number of observations
  #   k: Number of variables in w
  #   true_beta: True value of beta
  #
  # Returns:
  #   a list of 3 elements, y, x and w, where y and x are N 
  #   times 1 vectors, and w is an N times k matrix. 
  
  b=1/(1:k)
  
  # = Generate covariance matrix of w = #
  sigma=genPositiveDefMat(k,"unifcorrmat")$Sigma
  sigma=cov2cor(sigma)
  
  w=rmvnorm(N,sigma=sigma) # Generate w
  g=as.vector(cos(w%*%b)^2) # Generate the function g 
  m=as.vector(sin(w%*%b)+cos(w%*%b)) # Generate the function m 
  x=m+rnorm(N) # Generate x
  y=true_beta*x+g+rnorm(N) # Generate y
  
  dgp = list(y=y, x=x, w=w)
  
  return(dgp)
}

# Set seed ----

set.seed(12)



# Simulation of parameter beta distribution with OLS and LASSO ----

## OLS


ols_estimate_dist <- 1:100 %>% 
    map(function(x){
      tryCatch({
      print(x)
      dgp = generate_data()
      SL.library <- "SL.lm"
      sl_lm <- SuperLearner(Y = dgp$y,
                            X = data.frame(x=dgp$x, w=dgp$w), 
                            family = gaussian(),
                            SL.library = SL.library, 
                            cvControl = list(V=0))},
      error = function(e){
        cat(crayon::bgRed("Error: linear model not running!"))
      }
      )
      }
      ) %>% 
    map(~ coef(.x$fitLibrary$SL.lm_All$object)[2]) %>% 
    bind_rows(.id = "replication") %>% 
    rename(estimate = x)


ols_estimate_dist %>% 
  ggplot(aes(estimate)) +
  geom_density() +
  theme_minimal()

mean_estimator=c(mean(ols_estimate_dist$estimate))


## LASSO


lasso_estimate_dist <- 1:100 %>% 
  map(function(x){
    tryCatch({
      print(x)
      dgp = generate_data()
      SL.library <- "SL.glmnet"
      sl_lm <- SuperLearner(Y = dgp$y,
                            X = data.frame(x=dgp$x, w=dgp$w), 
                            family = gaussian(),
                            SL.library = SL.library, 
                            cvControl = list(V=0))},
      error = function(e){
        cat(crayon::bgRed("Error: LASSO model not running!"))
      }
    )
  }
  ) %>% 
  map(~ coef(.x$fitLibrary$SL.glmnet_All$object, s="lambda.min")[2]) %>%
  map(~ data.frame(estimate = .x)) %>% 
  bind_rows(.id = "replication") 

lasso_estimate_dist %>% 
  ggplot(aes(estimate)) +
  geom_density() +
  theme_minimal()


mean_estimator=c(mean_estimator, mean(lasso_estimate_dist$estimate))


# Double de-biased LASSO ----

double_lasso_dist <- 1:100 %>% 
  map(function(x){
    tryCatch({
      print(x)
      dgp = generate_data()
      
  SL.library <- lasso$names
  sl_lasso <- SuperLearner(Y = dgp$y,
                           X = data.frame(x=dgp$x, w=dgp$w), 
                           family = gaussian(),
                           SL.library = SL.library, 
                           cvControl = list(V=0))
  
  kept_variables <- which(get_lasso_coeffs(sl_lasso)!=0) - 1 # minus 1 as X is listed
  kept_variables <- kept_variables[kept_variables>0]
  
  sl_pred_x <- SuperLearner(Y = dgp$x,
                            X = data.frame(w=dgp$w), 
                            family = gaussian(),
                            SL.library = lasso$names, cvControl = list(V=0))
  
  kept_variables2 <- which(get_lasso_coeffs(sl_pred_x)!=0) 
  kept_variables2 <- kept_variables2[kept_variables2>0]
  
  SuperLearner(Y = dgp$y,
               X = data.frame(x = dgp$x, w = dgp$w[, c(kept_variables, kept_variables2)]), 
                                     family = gaussian(),
                                     SL.library = "SL.lm", 
                                     cvControl = list(V=0))},
  error = function(e){
    cat(crayon::bgRed("Error: double debiasing not running!"))
  })
  }) %>%
  map(~ coef(.x$fitLibrary$SL.lm_All$object)[2]) %>% 
  bind_rows(.id = "replication") %>% 
  rename(estimate = x) 
  
  
  
double_lasso_dist %>% 
  ggplot(aes(estimate)) +
  geom_density() +
  theme_minimal()


mean_estimator=c(mean_estimator, mean(double_lasso_dist$estimate))


# Naive Frisch-Waugh + ML Method of Choice ----


naive_dist <- 1:10 %>% 
  map(function(x){
    tryCatch({
      print(x)
      dgp = generate_data()
  
  sl_x = SuperLearner(Y = dgp$x, 
                      X = data.frame(w=dgp$w), # the data used to train the model
                      newX= data.frame(w=dgp$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  x_hat <- sl_x$SL.predict
  sl_y = SuperLearner(Y = dgp$y, 
                      X = data.frame(w=dgp$w), # the data used to train the model
                      newX= data.frame(w=dgp$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
    y_hat <- sl_y$SL.predict
    res_x = dgp$x - x_hat
    res_y = dgp$y - y_hat
    beta = (mean(res_x*res_y))/(mean(res_x**2))
    return(beta)}, # (coefficient of regression of res_y on res_x)
  error = function(e){
    cat(crayon::bgRed("Frisch-Waugh not running!"))
  })
  }) %>% 
  map(~ data.frame(estimate = .x)) %>% 
  bind_rows(.id = "replication") 


mean_estimator=c(mean_estimator, mean(naive_dist$estimate))

# Fritsch-Waugh Theorem does not work in non-linear cases!!


# Removing bias ----

removing_bias_dist <- 1:10 %>% 
  map(function(x){
    tryCatch({
      print(x)
      dgp = generate_data()
      
  split <- sample(seq_len(length(dgp$y)), size = ceiling(length(dgp$y)/2))
  
  dgp1 = list(y = dgp$y[split], x = dgp$x[split], w = dgp$w[split,])
  dgp2 = list(y = dgp$y[-split], x = dgp$x[-split], w = dgp$w[-split,])
  
  sl_x = SuperLearner(Y = dgp1$x, 
                      X = data.frame(w=dgp1$w), # the data used to train the model
                      newX= data.frame(w=dgp1$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  x_hat <- predict(sl_x, dgp2$w)$pred
  
  sl_y = SuperLearner(Y = dgp1$y, 
                      X = data.frame(w=dgp1$w), # the data used to train the model
                      newX= data.frame(w=dgp1$w), # the data used to predict x
                      family = gaussian(), 
                      SL.library = "SL.xgboost", # use whatever ML technique you like
                      cvControl = list(V=0)) 
  
  y_hat <- predict(sl_y, dgp2$w)$pred
  
  res_x = dgp2$x - x_hat
  res_y = dgp2$y - y_hat
  beta = (mean(res_x*res_y))/(mean(res_x**2)) # (coefficient of regression of res_y on res_x)
  return(beta)},
  error = cat(crayon::bgRed("Correction of Frisch-Waugh not running!"))
    )}
  ) %>%
  map(~ data.frame(estimate = .x)) %>% 
  bind_rows(.id = "replication") 
    


mean_estimator=c(mean_estimator, mean(removing_bias_dist$estimate))



# Double Machine Learning -----

doubleml <- function(X, W, Y, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost", family.X = gaussian(), family.Y = gaussian()) {
  
  ### STEP 1: split X, W and Y into 2 random sets (done for you)
  split <- sample(seq_len(length(Y)), size = ceiling(length(Y)/2))
  
  Y1 = Y[split]
  Y2 = Y[-split]
  
  X1 = X[split]
  X2 = X[-split]
  
  W1 = W[split, ]
  W2 = W[-split, ]
  
### STEP 2a: use a SuperLearner to train a model for E[X|W] on set 1 and predict X on set 2 using this model. Do the same but training on set 2 and predicting on set 1

  fitted_x <- 1:2 %>% 
  map(~ SuperLearner(Y = get(paste0("X",.x)), 
                    X = data.frame(get(paste0("W",.x))), 
                    family = family.X, 
                    SL.library = SL.library.X,
                    cvControl = list(V=0))
  ) %>% 
    map2(2:1, ~ predict(.x, get(paste0("W",.y)))$pred) %>% 
    map(~ .x[,1])
  
  ### STEP 2b: get the residuals X - X_hat on set 2 and on set 1

  residuals_x = fitted_x %>% 
    map2(list(X2,X1), ~ .y - .x)
  
  ### STEP 3a: use a SuperLearner to train a model for E[Y|W] on set 1 and predict Y on set 2 using this model. Do the same but training on set 2 and predicting on set 1
  
  fitted_y <- 1:2 %>% 
    map(~ SuperLearner(Y = get(paste0("Y",.x)), 
                       X = data.frame(get(paste0("W",.x))), 
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0))
    ) %>% 
    map2(2:1, ~ predict(.x, get(paste0("W",.y)))$pred) %>% 
    map(~ .x[,1])
  
  
  ### STEP 3b: get the residuals Y - Y_hat on set 2 and on set 1
  
  residuals_y = fitted_y %>% 
    map2(list(Y2,Y1), ~ .y - .x)
  
  
  ### STEP 4: regress (Y - Y_hat) on (X - X_hat) on set 1 and on set 2, and get the coefficients of (X - X_hat)
  
  beta_df <- residuals_y %>% 
    map2(residuals_x, ~ SuperLearner(Y = .x, 
                       X = data.frame(.y), 
                       family = family.Y, 
                       SL.library = "SL.lm",
                       cvControl = list(V=0))) %>% 
    map(~ coef(.x$fitLibrary$SL.lm_All$object)[2]) %>% 
    map(~ data.frame(coefficient = .x)) %>% 
    bind_rows()
  
  
  ### STEP 5: take the average of these 2 coefficients from the 2 sets (= beta)


  beta=mean(beta_df$coefficient,na.rm = T)
  ### STEP 6: compute standard errors (done for you). This is just the usual OLS standard errors in the regression res_y = res_x*beta + eps.
  
  

  psi_stack = c((residuals_y[[1]] - residuals_x[[1]]*beta_df[1,1]), (residuals_y[[2]] - residuals_x[[2]]*beta_df[[2,1]]))
  res_stack = c(residuals_x[[1]], residuals_x[[2]])
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(Y))
  
  return(c(beta = beta, se = se))
}


# Run double machine learning simulation -----


double_ml_dist <- 1:10 %>% 
  map(function(x){
    tryCatch({
    print(x)
    dgp = generate_data()
    
    doubleml(dgp$x,dgp$w,dgp$y)
    },
    error = function(e){
      cat(crayon::bgRed("Double ML not working!"))
    })
  }) %>% 
  map(~ data.frame(beta = .x[1])) %>% 
  bind_rows()



double_ml_dist %>% 
ggplot(aes(beta)) +
  geom_density() +
  theme_minimal()
