# 1) Pre-processing of text data 
# 2) Method
# 3) Hyperparameter tuning
# 4) Length of training data

# Install and load all required packages ----


packages=c("tidyverse","stopwords","tidytext",
           "reshape2","SuperLearner","corpus",
           "xgboost","tm", "SnowballC", "crayon",
           "ROCR")



packages %>% 
  lapply(function(x){
  do.call("require", list(x))
  }
)


# Load raw data ----

# Set parameters:

local_path=c("../Machine_learning_for_economics_material/raw_data/homework_1/")
name_files=c("test","train")


# Read csv files:

list_df <- name_files %>% 
  map_chr(~ paste0(local_path,.x,".csv")) %>% 
  map(~ read.csv(.x)) %>% 
  map(~ as_tibble(.x))

names(list_df) <- name_files


# Pre-processing tweets ----
# Note: single words, bigrams and single words stemmed

stpwds = stopwords::stopwords(source = "stopwords-iso") 
 
replace_reg <- "http[s]?://[A-Za-z\\d/\\.]+|&amp;|&lt;|&gt;"
unnest_reg  <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"


# Bi-gram 

clean_list_bigrams <- list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% mutate(Tweet = tolower(Tweet))) %>% 
  map(~ .x %>% mutate(Tweet = removeWords(.$Tweet,stpwds))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "ngrams", n= 2))

# Words 

clean_list_words <- list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "regex",pattern = unnest_reg, to_lower = T)) %>% 
  map(~ .x %>% filter(!word %in% stpwds & str_detect(word, "[a-z]")))


# Words (stemmed - Porter algorithm)

clean_list_stemmed <- list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "regex",pattern = unnest_reg, to_lower = T)) %>% 
  map(~ .x %>% filter(!word %in% stpwds & str_detect(word, "[a-z]"))) %>% 
  map(~ .x %>% mutate(word = wordStem(word))) 
  
  
# Compare stemming with single words tokenization
  
if(clean_list_stemmed[["train"]] %>%
    group_by(word) %>% count() %>% .$n %>% length() <= clean_list_words[["train"]] %>% group_by(word) %>% count() %>% .$n %>% length()){
  cat(green("Stemming reduces dimensionality!"))
} else{
  cat(red("Stemming does not reduce dimensionality"))
}
  
  

    
# Keep only 500 best words/stemmed words/n-grams -----

# Regroup different pre-processed test and train sets together: 

test <- rep(1,3) %>% 
  map2(list(clean_list_words, clean_list_bigrams,clean_list_stemmed), ~ .y[[.x]])

train <- rep(2,3) %>% 
  map2(list(clean_list_words, clean_list_bigrams,clean_list_stemmed), ~ .y[[.x]])

# Select only common words (train and test) and keep best 500:
    
list_of_words= test %>% 
  map2(train, ~ .y$word[.y$word %in% .x$word])

freq_words <- list_of_words %>% 
  map(~ data.frame(word=.x)) %>% 
  map(~ .x %>% group_by(word)) %>% 
  map(~ .x %>% mutate(n = n())) %>% 
  map(~ .x %>% unique()) %>% 
  map(~ .x %>% ungroup()) %>% 
  map(~ .x %>% arrange(-n)) %>% 
  map(~ .x %>% dplyr::slice(1:500)) %>% 
  map(~ .x %>% mutate(word = as.character(word)))
  
top_500 <- list(train,test) %>%
  flatten() %>% 
  map2(rep(freq_words,2), ~ .x %>% mutate(topwords = ifelse(word %in% .y$word, 1, 0))) %>% 
  map(~ .x %>% mutate(word = ifelse(topwords==1, word, "no_top_word"))) %>%
  map(~ .x %>% unique()) %>%
  map(~ .x %>% group_by(id)) %>%
  map(~ .x %>% mutate(notopwords = 1-max(topwords))) %>%
  map(~ .x %>% ungroup()) %>%
  map(~ .x %>% filter(!(word=="no_top_word" & notopwords==0))) %>%
  map(~ .x %>% select(-topwords, -notopwords)) %>%
  map(~ .x %>% unique()) 

  



tidy_tweets_topwords <- top_500 %>% 
  map(~ if(any(names(.x) == "Author")){
    .x %>% reshape2::dcast(id+Author~word, function(x) 1, fill = 0)
  } else {
    .x %>% reshape2::dcast(id~word, function(x) 1, fill = 0) 
  }
 )
  

    
# Save intermediate files ----

# Set parameters:

intermediate_path="../Machine_learning_for_economics_material/intermediate_data/homework_1/"   
name_intermediate=c(rep("train",3),rep("test",3))
pre_processing=rep(c("word","stemmed","bigram"), 2)

# Export:
 
  pwalk(list(tidy_tweets_topwords, 
            name_intermediate, 
            pre_processing), function(x,y,z){
             saveRDS(x, file = paste0(intermediate_path,y,"_",z,".rds"))
              }
       )

        
# Create a "test" set from already labelled data ---- 
# Note: seed goes inside the function otherwise randomness not common to iterations
  

# Partition the train and test set from labelled data (70% train)

label_train <- tidy_tweets_topwords[1:3] %>% 
  map(~ as_tibble(.x)) %>%
  map( function(x){
    set.seed(123)
    x %>% sample_frac(.7)}
    )
  
label_test <- tidy_tweets_topwords[1:3] %>% 
  map2(label_train, ~ .x %>% filter(!id %in% .y$id)) %>% 
  map(~ as_tibble(.x))

# Partition between x's and y's


label_train_x <- label_train %>% 
  map(~ .x %>% select(-id, -Author))

label_train_y <- label_train %>% 
  map(~ .x %>% select(Author)) %>% 
  map(~ .x %>% mutate(Author = case_when(Author == "bernie" ~ 1,
                      T~ 0)))


label_test_x <- label_test %>% 
  map(~ .x %>% select(-id, -Author))

label_test_y <- label_test %>% 
  map(~ .x %>% select(Author)) %>% 
  map(~ .x %>% mutate(Author = case_when(Author == "bernie" ~ 1,
                                         T~ 0)))
  
# Train the models ----
# Note: running three models for three pre-processing methods (may take up to three minutes)

set.seed(436)


model <- label_train_x %>% 
  map2(label_train_y, ~ SuperLearner(Y= .y$Author, 
                      X= .x,
                      family = binomial(),
                      SL.library = c("SL.mean",
                                     "SL.kernelKnn",
                                     "SL.glmnet"),
                      cvControl = list(0)))


# Obtain fitted probabilities for labelled "test" set:

fitted <- model %>% 
    map2(label_test_x, ~ predict(.x, .y, onlySL = F)[["library.predict"]]) %>% 
    map(~ as_tibble(.x))


# Rank models by Area under the Curve and plot:

name_models=c("Uncond. mean","KNN","LASSO")
token=c(rep("Words", 3),rep("Bi-gram",3),rep("Stemmed Words",3))

auc_df <- fitted %>% 
  map2(label_test_y, ~ cbind(.x,.y)) %>% 
  map(~ map(1:3, function(x){
    ROCR::prediction(.x[,x],.x$Author)
  })) %>% 
  modify_depth(2,~ ROCR::performance(.x, measure = "auc", x.measure = "cutoff")@y.values[[1]]) %>% 
  modify_depth(2,~ data.frame(auc = .x)) %>% 
  map(~ bind_rows(.x)) %>% 
  map(~ .x %>% mutate(models = name_models)) %>% 
  bind_rows() %>% 
  mutate(pre_processing = token)
  

auc_df %>%
  mutate(models = factor(models)) %>% 
  mutate(models = fct_reorder(models,auc, mean)) %>% 
  ggplot(aes(auc, models)) +
  geom_col(width = 0.2) +
  facet_wrap(~ pre_processing) +
  theme_minimal() +
  xlab("Area under the Curve (AUC)") +
  ylab("Algorithm")




# Extend on "real" test set ----


predict(model, tidy_tweets_topwords_test %>% select(-id))$pred %>% 
  round(.,2) %>% 
  data.frame(fitted = .) %>% 
  ggplot(aes(fitted)) +
  geom_density() +
  theme_minimal()





  
# Grid search hyperparameters: ---- 
  
learner = create.Learner("SL.kernelKnn", tune = list(k = c(2:8)))

cv_sl = SuperLearner(Y= ifelse(tidy_tweets_topwords_train$Author == "bernie",1,0), 
                     X= tidy_tweets_topwords_train %>% select(-id,-Author),
                     family = binomial(),
                     SL.library = learner$names,
                     cvControl = list(5))
  
cv_sl$cvRisk %>% 
  stack() %>% 
  mutate(ind = str_extract(ind, "\\d")) %>%
  slice(1:9) %>% 
  setNames(c("Risk","K")) %>% 
  ggplot(aes(K, Risk, group = 1)) +
  geom_line(col = "blue", size = 1.5) +
  geom_point(col = "blue", alpha = 0.5, size = 3) +
  theme_minimal()
  
    
    
