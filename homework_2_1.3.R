###### INSTRUCTIONS TO RUN -----

# Change data_path to path of local raw data files
# Change export_path_tables to path where to export output tables
# Running time: approx. ...

######

# Load all required packages -----

packages= c("tidyverse","SuperLearner", "clusterGeneration", "mvtnorm", "xgboost",
            "stargazer")

lapply(packages, function(x){
  do.call("require", list(x))
}
)


# Custom function for Double ML -----


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



# Read the data ----


data_path = c("../Machine_learning_for_economics_material/raw_data/homework_2/social_turnout.csv")

df <-  read.csv(data_path) %>% 
  as_tibble() 

# Double ML ----

set.seed(245)

doubleml <- doubleml(df$treat_neighbors, df %>% dplyr::select(-outcome_voted,-treat_neighbors) %>% as.matrix(), df$outcome_voted,
     family.X = binomial(), family.Y = binomial()) %>% 
  stack() %>% 
  spread(ind,values) %>% 
  setNames(c("Estimate","Std. Error")) %>% 
  mutate(Estimation = "Double ML (XGBoost)")

# OLS without controls ----

lm_no_controls <- lm(outcome_voted ~ treat_neighbors, df) %>% 
  summary() %>% 
  coef() %>%
  data.frame() %>% 
  dplyr::slice(2) %>% 
  rename(`Std. Error` = `Std..Error`) %>% 
  dplyr::select(Estimate, `Std. Error`) %>% 
  mutate(Estimation = "OLS with no controls")


# OLS with controls ----


controls <- df %>% 
  names() %>% 
  str_subset(.,"treat_neighbors",negate = T) %>% 
  str_subset(.,"outcome_voted", negate = T) %>% 
  paste0(., collapse = " + ")


formula=paste0("outcome_voted ~ treat_neighbors + ", name_controls)

lm_controls <- lm(formula, df) %>%
  summary() %>% 
  coef() %>%
  data.frame() %>% 
  dplyr::slice(2) %>% 
  rename(`Std. Error` = `Std..Error`) %>% 
  dplyr::select(Estimate, `Std. Error`) %>% 
  mutate(Estimation = "OLS with controls")




# Bind results together and export: -----

export_path_tables="../Machine_learning_for_economics_material/output/homework_2/tables/"

rbind(lm_no_controls, lm_controls, doubleml) %>%
  dplyr::select(Estimation, Estimate, `Std. Error`) %>%
  mutate(`95% - Lower` = Estimate - 1.96*`Std. Error`,
         `95% - Upper `= Estimate + 1.96*`Std. Error`) %>% 
  stargazer(summary = F,
            rownames = F,
            out = paste0(export_path_tables,"comparison_voter_estimation.tex"))






         
         
         
         