# Custom function for Double ML -----


k2ml <- function(X, W, Y, K=5, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost", family.X = gaussian(), family.Y = gaussian()) {
  
  
  #' Applies double ML (Chernozukov et al., 2018) to treatment estimation problems
  #' 
  #' @param X numeric vector with treatment variable
  #' @param W matrix with control variables 
  #' @param Y numeric vector with outcome variable
  #' @param K integer. Number of samples splitting deployed in double ML
  #' @param Sl.library.X character string. Algorithm for SuperLearner used in predicting treatment variable with controls
  #' @param Sl.library.T character string. Algorithm for SuperLearner used in predicting outcome variable with controls
  #' @param family.X function. Either gaussian (continous treatment) or binomial (discrete treatment)
  #' @param family.Y function. Either gaussian (continous outcome) or binomial (discrete treatment)
  #' 
  #' @return named vector with estimated treatment coefficient (average of k-splitting) and associated standard error
  
  # Passage required only for automated script homework - for external use comment!
  
  install.packages("caret")
  library(caret)
  
  ### STEP 1: split X,Y and W in k-folds
  
  split_index = X %>% createFolds(K)
  
  X1 <- split_index %>% 
    map(~ X[-.x])
  X2 <- split_index %>% 
    map(~ X[.x]) 
  
  Y1 <- split_index %>% 
    map(~ Y[-.x])
  Y2 <- split_index %>% 
    map(~ Y[.x]) 
  
  W1 <- split_index %>% 
    map(~ W[-.x,])
  W2 <- split_index %>% 
    map(~ W[.x,]) 
  

  ### STEP 2a: use a SuperLearner to train a model for E[X|W] on k-sets and predict X on the respective k-fold. 
  
  fitted_x <- 1:K %>% 
    map(~ SuperLearner(Y = X1[[.x]], 
                       X = data.frame(W1[[.x]]), 
                       family = family.X, 
                       SL.library = SL.library.X,
                       cvControl = list(V=0))
    ) %>% 
    map2(W2, ~ predict(.x, .y)$pred) %>% 
    map(~ .x[,1])
  
  ### STEP 2b: get the residuals X - X_hat 
  
  residuals_x = fitted_x %>% 
    map2(X2, ~ .y - .x)
  
  
  ### STEP 3a: use a SuperLearner to train a model for E[Y|W] on k-sets and predict Y on the respective k-fold. 
  fitted_y <- 1:K %>% 
    map(~ SuperLearner(Y = Y1[[.x]], 
                       X = data.frame(W1[[.x]]), 
                       family = family.Y, 
                       SL.library = SL.library.Y,
                       cvControl = list(V=0))
    ) %>% 
    map2(W2, ~ predict(.x, .y)$pred) %>% 
    map(~ .x[,1])
  
  
  
  ### STEP 3b: get the residuals Y - Y_hat 
  
  residuals_y = fitted_y %>% 
    map2(Y2, ~ .y - .x)
  
  
  ### STEP 4: regress (Y - Y_hat) on (X - X_hat) and get the coefficients of (X - X_hat)
  
  beta_df <- residuals_y %>% 
    map2(residuals_x, ~ SuperLearner(Y = .x, 
                                     X = data.frame(.y), 
                                     family = family.Y, 
                                     SL.library = "SL.lm",
                                     cvControl = list(V=0))) %>% 
    map(~ coef(.x$fitLibrary$SL.lm_All$object)[2]) %>% 
    map(~ data.frame(coefficient = .x)) %>% 
    bind_rows()
  
  ### STEP 5: take the average of these k coefficients
  
  
  beta=mean(beta_df$coefficient,na.rm = T)
  
  
  ### STEP 6: compute standard errors (done for you). This is just the usual OLS standard errors in the regression res_y = res_x*beta + eps.
  
  psi_stack = 1:K %>% 
    map(~ residuals_y[[K]] - residuals_x[[K]]*beta_df[K,1]) %>% 
    unlist()
  
  res_stack = 1:K %>% 
    map(~ residuals_x[[K]]) %>% 
    unlist()
  
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(Y))
  
  # Return treatment coefficient and standard error:
  
  return(c(beta = beta, se = se))
  
}




  
