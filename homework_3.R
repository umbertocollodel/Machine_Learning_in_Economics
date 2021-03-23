############ INSTRUCTIONS TO RUN ###############

# Change raw_path to repository in local machine with survey data in .rds
# Change export_path_figures to repository for export output figures


# Script estimated running time: about 3.5 minutes 

##################################################

# Install and load all required packages ----


packages=c("tidyverse", "devtools","caret",
           "SuperLearner")



lapply(packages, function(x){
  do.call("require", list(x))
}
)

# install causalTree package directly from .tar file

path_pkg = "~/Downloads/causalTree_0.0.tar"

install.packages(path_pkg, repos = NULL, type = "source")

library(causalTree)


# Applying Causal Trees ----


set.seed(345)

# Read data: we keep only two-thirds of the original observations to reduce 
# computational time

raw_path="../Machine_learning_for_economics_material/raw_data/homework 3/welfare.rds"

df=readRDS(raw_path) 


# Split into two random samples:

index=sample(1:nrow(df),nrow(df)/2)


df1=df[index,]
df2=df[-index,]


# Get formula

names_independent <- df %>% 
  names() %>% 
  str_subset(.,"^y$",negate = T)

tree_fml=as.formula(paste("y", paste(names_independent, collapse = ' + '), sep = " ~ "))

# Run causal tree and plot result:

causal_tree <- causalTree::causalTree(formula = tree_fml,
                          data = df,
                          treatment = df$w,
                          split.Rule = "CT", #causal tree
                          split.Honest = F, #will talk about this next
                          split.alpha = 1, #will talk about this next
                          cv.option = "CT",
                          cv.Honest = F,
                          split.Bucket = T, #each bucket contains bucketNum treated and bucketNum control units
                          bucketNum = 5, 
                          bucketMax = 100, 
                          minsize = 250) # number of observations in treatment and control on leaf

rpart.plot(causal_tree, roundint = F)


# Tree too deep and complex: prune with optimal cross-validation cp


optimal_cp = causal_tree$cptable[which.min(causal_tree$cptable[,4]),1]

pruned_tree <- prune(causal_tree, optimal_cp)
rpart.plot(pruned_tree,
           type = 3,
           clip.right.labs = F,
           branch = F,
           roundint = F)

# Export:

export_path_figures="../Machine_learning_for_economics_material/output/homework_3/figures/"

ggsave(paste0(export_path,"causal_tree_pruned.pdf"))




# Applying Best Linear Predictor -----

# Create dataframe with no missing controls in observations:


df_na_clean <- df %>% 
  filter(complete.cases(.))


# Custom functions:

zeros <- function(n) {
  return(integer(n))
}
ones <- function(n) {
  return(integer(n)+1)
}


blp <- function(Y, W, X, prop_scores=F) {
  
  ### STEP 1: split the dataset into two sets, 1 and 2 (50/50)
  split <- createFolds(1:length(Y), k=2)[[1]]
  
  Ya = Y[split]
  Yb = Y[-split]
  
  Xa = X[split]
  Xb = X[-split]
  
  Wa = W[split, ]
  Wb = W[-split, ]
  
  
  ### STEP 2a: (Propensity score) On set A, train a model to predict X using W. Predict on set B.
  if (prop_scores==T) {
    sl_w1 = SuperLearner(Y = Xa, 
                         X = Wa, 
                         newX = Wb, 
                         family = binomial(), 
                         SL.library = "SL.xgboost", 
                         cvControl = list(V=0))
    
    p <- sl_w1$SL.predict
  } else {
    p <- rep(mean(Xa), length(Xb))
  }
  ### STEP 2b let D = W(set B) - propensity score.
  D <- Xb-p
  
  ### STEP 3a: Get CATE (for example using xgboost) on set A. Predict on set B.
  sl_y = SuperLearner(Y = Ya, 
                      X = data.frame(X=Xa, Wa), 
                      family = gaussian(), 
                      SL.library = "SL.xgboost", 
                      cvControl = list(V=0))
  

  pred_y1 = predict(sl_y, newdata=data.frame(X=ones(nrow(Wb)), Wb))
  
  pred_0s <- predict(sl_y, data.frame(X=zeros(nrow(Wb)), Wb), onlySL = T)
  pred_1s <- predict(sl_y, data.frame(X=ones(nrow(Wb)), Wb), onlySL = T)
  
  cate <- pred_1s$pred - pred_0s$pred
  
  ### STEP 3b: Subtract the expected CATE from the CATE
  C = cate-mean(cate)
  
  ### STEP 4: Create a dataframe with Y, W (set B), D, C and p. Regress Y on W, D and D*C. 
  df <- data.frame(Y=Yb, Wb, D, C, p)
  
  Wnames <- paste(colnames(Wb), collapse="+")
  fml <- paste("Y ~",Wnames,"+ D + D:C")
  model <- lm(fml, df, weights = 1/(p*(1-p))) 
  
  return(model) 
}


table_from_blp <-function(model) {
  thetahat <- model%>% 
    .$coefficients %>%
    .[c("D","D:C")]
  
  # Confidence intervals
  cihat <- confint(model)[c("D","D:C"),]
  
  res <- tibble(coefficient = c("beta1","beta2"),
                estimates = thetahat,
                ci_lower_90 = cihat[,1],
                ci_upper_90 = cihat[,2])
  
  return(res)
}



# Run BLP:

output <- rerun(10, table_from_blp(blp(df_na_clean$y, df_na_clean %>% select(-w,-y), df_na_clean$w))) %>% 
  bind_rows %>%
  group_by(coefficient) %>%
  summarize_all(median)


# Plot results and export:


output %>% 
  ggplot(aes(x = coefficient,ymin = ci_lower_90, ymax = ci_upper_90, col = coefficient)) +
  geom_errorbar(size = 1.5, width = 0.4) +
  labs(col = "") +
  xlab("") +
  ylim(-1,1) +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 16)) +
  theme(axis.text.x = element_blank(),
        axis.text = element_text(size = 20)) +
  theme(panel.grid.major.x = element_blank())
  

ggsave(paste0(export_path_figures,"blp_coefficients.pdf"))


# Applying GATES ----

# Custom functions:


gates <- function(Y, W, X, Q=4, prop_scores=F) {
  
  ### STEP 1: split the dataset into two sets, 1 and 2 (50/50)
  split <- createFolds(1:length(Y), k=2)[[1]]
  
  Ya = Y[split]
  Yb = Y[-split]
  
  Xa = X[split]
  Xb = X[-split]
  
  Wa = W[split, ]
  Wb = W[-split, ]
  
  ### STEP 2a: (Propensity score) On set A, train a model to predict X using W. Predict on set B.
  if (prop_scores==T) {
    sl_w1 = SuperLearner(Y = Xa, 
                         X = Wa, 
                         newX = Wb, 
                         family = binomial(), 
                         SL.library = "SL.xgboost", 
                         cvControl = list(V=0))
    
    p <- sl_w1$SL.predict
  } else {
    p <- rep(mean(Xa), length(Xb))
  }
  
  ### STEP 2b let D = W(set B) - propensity score.
  D <- Xb-p
  
  ### STEP 3a: Get CATE (for example using xgboost) on set A. Predict on set B.
  sl_y = SuperLearner(Y = Ya, 
                      X = data.frame(X=Xa, Wa), 
                      family = gaussian(), 
                      SL.library = "SL.xgboost", 
                      cvControl = list(V=0))
  
  pred_y1 = predict(sl_y, newdata=data.frame(X=ones(nrow(Wb)), Wb))
  
  pred_0s <- predict(sl_y, data.frame(X=zeros(nrow(Wb)), Wb), onlySL = T)
  pred_1s <- predict(sl_y, data.frame(X=ones(nrow(Wb)), Wb), onlySL = T)
  
  cate <- pred_1s$pred - pred_0s$pred
  
  ### STEP 3b: divide the cate estimates into Q tiles, and call this object G. 
  # Divide observations into n tiles
  G <- data.frame(cate) %>% # replace cate with the name of your predictions object
    ntile(Q) %>%  # Divide observations into Q-tiles
    factor()
  
  ### STEP 4: Create a dataframe with Y, W (set B), D, G and p. Regress Y on group membership variables and covariates. 
  df <- data.frame(Y=Yb, Wb, D, G, p)
  
  Wnames <- paste(colnames(Wb), collapse="+")
  fml <- paste("Y ~",Wnames,"+ D:G")
  model <- lm(fml, df, weights = 1/(p*(1-p))) 
  
  return(model) 
}


table_from_gates <-function(model) {
  thetahat <- model%>% 
    .$coefficients %>%
    .[c("D:G1","D:G2","D:G3","D:G4")]
  
  # Confidence intervals
  cihat <- confint(model)[c("D:G1","D:G2","D:G3","D:G4"),]
  
  res <- tibble(coefficient = c("gamma1","gamma2","gamma3","gamma4"),
                estimates = thetahat,
                ci_lower_90 = cihat[,1],
                ci_upper_90 = cihat[,2])
  
  return(res)
}



# Run GATES:


output <- rerun(10, table_from_gates(gates(df_na_clean$y, df_na_clean %>% select(-w,-y), df_na_clean$w))) %>% 
  bind_rows %>%
  group_by(coefficient) %>%
  summarize_all(median)



# Plot results and export:


output %>% 
  ggplot(aes(x = coefficient, ymin = ci_lower_90, ymax = ci_upper_90, col = coefficient)) +
  geom_errorbar(size = 1.5, width = 0.4) +
  ylim(-0.6,0.6) +
  xlab("") +
  labs(col="") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 16)) +
  theme(axis.text = element_text(size = 20),
        axis.text.x = element_blank()) +
  theme(panel.grid.major.x = element_blank())



ggsave(paste0(export_path_figures,"gates_coefficients.pdf"))

  