###### INSTRUCTIONS TO RUN -----

# Change data_path to path of local raw data files

######

# Load all required packages -----




# Read the data ----


data_path = c("../Machine_learning_for_economics_material/raw_data/homework_2/social_turnout.csv")

df <-  read.csv(data_path) %>% 
  as_tibble() 

# Double ML ----


k2ml(df$treat_neighbors, df %>% select(-outcome_voted,-treat_neighbors) %>% as.matrix(), df$outcome_voted,
     family.X = binomial(), family.Y = binomial())

# OLS without controls ----

lm_no_controls <- lm(outcome_voted ~ treat_neighbors, df) %>% 
  summary() %>% 
  coef() %>%
  data.frame() %>% 
  slice(2)


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
  slice(2)
  
