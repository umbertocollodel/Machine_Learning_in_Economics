library(caret)


# Extending the double ML function to the K-fold case instead of double splitting



# Double Machine Learning -----

doubleml <- function(X, W, Y, K=5, SL.library.X = "SL.xgboost",  SL.library.Y = "SL.xgboost", family.X = gaussian(), family.Y = gaussian()) {
  
  ### STEP 1: split X,Y and W in k-folds
  
  set.seed(143)
  
  split_index = X %>% createFolds(K)
  
  X1 <- split_index %>% 
    map(~ X[.x])
  X2 <- split_index %>% 
    map(~ X[-.x]) 
  
  Y1 <- split_index %>% 
    map(~ Y[.x])
  Y2 <- split_index %>% 
    map(~ Y[-.x]) 
  
  W1 <- split_index %>% 
    map(~ W[.x,])
  W2 <- split_index %>% 
    map(~ W[-.x,]) 
  

# Maybe find a more concise way to do it! without repetitions

  
  
  split <- sample(seq_len(length(Y)), size = ceiling(length(Y)/2))
  
  Y1 = Y[split]
  Y2 = Y[-split]
  
  X1 = X[split]
  X2 = X[-split]
  
  W1 = W[split, ]
  W2 = W[-split, ]
  
  
  return()
  
}
  
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




doubleml(dgp$x, dgp$w, dgp$y)



