###### INSTRUCTIONS TO RUN -----

# Change data_path to path of local raw data files
# Running time: approx. ...

######

# Load all required packages -----

packages= c("tidyverse","SuperLearner", "clusterGeneration", "mvtnorm", "xgboost",
            "stargazer")

lapply(packages, function(x){
  do.call("require", list(x))
}
)


# Read the data ----


data_path = c("../Machine_learning_for_economics_material/raw_data/homework_2/social_turnout.csv")

df <-  read.csv(data_path) %>% 
  as_tibble() 

# Double ML ----

set.seed(245)

doubleml <- doubleml(df$treat_neighbors, df %>% select(-outcome_voted,-treat_neighbors) %>% as.matrix(), df$outcome_voted,
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
  slice(2) %>% 
  rename(`Std. Error` = `Std..Error`) %>% 
  select(Estimate, `Std. Error`) %>% 
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
  slice(2) %>% 
  rename(`Std. Error` = `Std..Error`) %>% 
  select(Estimate, `Std. Error`) %>% 
  mutate(Estimation = "OLS with controls")




# Bind results together and export: -----

export_path_tables="../Machine_learning_for_economics_material/output/homework_2/tables/"

rbind(lm_no_controls, lm_controls, doubleml) %>%
  select(estimation, Estimate, `Std. Error`) %>%
  mutate(`95% - Lower` = Estimate - 1.96*`Std. Error`,
         `95% - Upper `= Estimate + 1.96*`Std. Error`) %>% 
  stargazer(summary = F,
            rownames = F,
            out = paste0(export_path_tables,"comparison_voter_estimation.tex"))






         
         
         
         