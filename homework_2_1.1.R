# Installs packages if not already installed, then loads packages -----
list.of.packages <- c("SuperLearner", "ggplot2", "glmnet", "clusterGeneration", "mvtnorm", "xgboost",
                      "crayon")
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



# Simulation of parameter beta distribution with OLS and LASSO ----

## OLS

beta_X = c()

ols_estimate_dist <- 1:250 %>% 
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


beta_X = c()
for (i in 1:200) {
  print(i)
  dgp = generate_data()
  
  SL.library <- "SL.glmnet" 
    sl_lasso <- SuperLearner(Y = dgp$y,
                             X = data.frame(x=dgp$x, w=dgp$w), 
                             family = gaussian(),
                             SL.library = SL.library, 
                             cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_lasso$fitLibrary$SL.glmnet_All$object, s="lambda.min")[2])
}

beta_X_df <- data.frame(beta_X=beta_X)

ggplot(beta_X_df, aes(x = beta_X)) + 
  geom_histogram(binwidth = 0.02) +
  theme_minimal()

mean_estimator=c(mean_estimator, mean(beta_X_df$beta_X))


# Double de-biased LASSO ----

beta_X = c()
for (i in 1:100) {
  print(i)
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
  
  sl_screening_lasso <- SuperLearner(Y = dgp$y,
                                     X = data.frame(x = dgp$x, w = dgp$w[, c(kept_variables, kept_variables2)]), 
                                     family = gaussian(),
                                     SL.library = "SL.lm", 
                                     cvControl = list(V=0))
  
  beta_X = c(beta_X, coef(sl_screening_lasso$fitLibrary$SL.lm_All$object)[2])
}

beta_X_df <- data.frame(beta_X=beta_X)
ggplot(beta_X_df, aes(x = beta_X)) + geom_histogram(binwidth = 0.02)

mean_estimator=c(mean_estimator, mean(beta_X_df$beta_X))


# Naive Frisch-Waugh + ML Method of Choice


beta_X = c()
for (i in 1:30) {
  print(i)
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
    beta = (mean(res_x*res_y))/(mean(res_x**2)) # (coefficient of regression of res_y on res_x)
  beta_X = c(beta_X, beta)
}

beta_X_df <- data.frame(beta_X=beta_X)
ggplot(beta_X_df, aes(x = beta_X)) + geom_histogram(binwidth = 0.02)


mean(beta_X_df$beta_X)

# Fritsch-Waugh Theorem does not work in non-linear cases!!


# Removing bias ----

beta_X = c()
for (i in 1:30) {
  print(i)
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
  beta_X = c(beta_X, beta)
  }

beta_X_df <- data.frame(beta_X=beta_X)
ggplot(beta_X_df, aes(x = beta_X)) + geom_histogram(binwidth = 0.02)

mean(beta_X_df$beta_X)






