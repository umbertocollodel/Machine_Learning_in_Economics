############ INSTRUCTIONS TO RUN ###############


#
# Script estimated running time: about 3.5 minutes 

##################################################

# Install and load all required packages ----


packages=c("tidyverse", "devtools")



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

df=readRDS(raw_path) %>% 
  slice(sample(nrow(.),nrow(.)*2/3))


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






