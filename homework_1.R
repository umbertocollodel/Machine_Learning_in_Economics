# 1) Pre-processing of text data 
# 2) Method
# 3) Hyperparameter tuning
# 4) Length of training data


library(tidyverse)
library(stopwords)
library(tidytext)
library(reshape2)
library(SuperLearner)
library(tidymodels)
library(corpus)
library(xgboost)
library(future)




# Load raw data ----

name_files=c("test","train")


list_df <- name_files %>% 
  map_chr(~ paste0("../Machine_learning_for_economics_material/raw_data/homework_1/",.x,".csv")) %>% 
  map(~ read.csv(.x)) %>% 
  map(~ as_tibble(.x))

names(list_df) <- name_files


# Pre-processing ----
# Note: single words, bigrams and single words stemmed

stpwds = stopwords::stopwords(source = "stopwords-iso") 
 
replace_reg <- "http[s]?://[A-Za-z\\d/\\.]+|&amp;|&lt;|&gt;"
unnest_reg  <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"


# Bi-gram ----

clean_list_df <- list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% mutate(Tweet = tolower(Tweet))) %>% 
  map(~ .x %>% mutate(Tweet = removeWords(.$Tweet,stpwds))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "ngrams", n= 2))

# Words ----

clean_list_df <- list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "regex",pattern = unnest_reg, to_lower = T)) %>% 
  map(~ .x %>% filter(!word %in% stpwds & str_detect(word, "[a-z]")))


# Words (stemmed) ----


list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "regex",pattern = unnest_reg, to_lower = T)) %>% 
  map(~ .x %>% filter(!word %in% stpwds & str_detect(word, "[a-z]")))


# still to work on this!!




# Continue to work on plot:

clean_list_df[["train"]] %>%
  filter(word != "rt") %>% 
  group_by(Author,word) %>% 
  count() %>% 
  filter(n <100) %>% 
  spread(Author, n) %>% 
  ggplot(aes(bernie,trump)) +
    geom_point() +
    geom_label(aes(label = ))
    theme_minimal()

    
#############################
##### GET TOP 500 WORDS #####
#############################
    
## in this part we want to create a dataframe with a list of tweet ids and dummy variables for the top 500 words
    
tidy_tweets_train <- clean_list_df[["train"]]
tidy_tweets_test <- clean_list_df[["test"]]
    
    
list_of_words = tidy_tweets_train$word[tidy_tweets_train$word %in% tidy_tweets_test$word]
    
freq_words <- data.frame(word=list_of_words) %>%
      group_by(word) %>%
      mutate(n = n()) %>%
      unique() %>%
      ungroup() %>%
      arrange(-n) %>%
      .[1:500,] %>% # top 500 words
      mutate(word=as.character(word))
    
top_500 <- list(tidy_tweets_train, tidy_tweets_test) %>%
  map(~ .x %>% mutate(topwords = ifelse(word %in% freq_words$word, 1,0))) %>%
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
    
tidy_tweets_topwords %>% 
      walk2(name_files, ~ saveRDS(.x, file = paste0("../Machine_learning_for_economics_material/intermediate_data/homework_1/",.y,"_top500_words.rds")))
 
 
# Create a training sample and validation sample of the insample tweets

 set.seed(123)

 
train <- tidy_tweets_topwords_train %>% 
  sample_frac(.7) %>% 
  as_tibble()

test <- tidy_tweets_topwords_train %>% 
  filter(!id %in% train$id) %>% 
  as_tibble()

train_x <- train %>% 
  select(-id, -Author)

train_y <- train %>% 
  select(Author) %>% 
  mutate(Author = case_when(Author == "bernie" ~ 1,
                            T~ 0))

test_x <- test %>% 
  select(-id, -Author)

test_y <- test %>%
  select(Author) %>% 
  mutate(Author = case_when(Author == "bernie" ~ 1,
                            T~ 0))
 
  
# Train the model ----

set.seed(123)

model <- SuperLearner(Y= train_y$Author, 
               X= train_x,
               family = binomial(),
               SL.library = c("SL.mean",
                              "SL.kernelKnn",
                              "SL.glmnet",
                              "SL.randomForest",
                              "SL.xgboost"),
               cvControl = list(0))


predicted <- predict(model, test_x, onlySL = F)[["library.predict"]] %>% 
  as_tibble()

names(predicted) %>% 
  map(~ ROCR::prediction(predicted[,.x], test_y$Author)) %>%
  map(~ ROCR::performance(.x, measure = "auc", x.measure = "cutoff")@y.values[[1]]) %>% 
  map(~ data.frame(auc = .x)) %>% 
  bind_rows() %>% 
  mutate(model = names(predicted)) %>% 
  mutate(model = str_extract(model, "(?<=\\.)(.+)(?=_)")) %>%
  mutate(model = fct_reorder(model, auc, mean)) %>% 
  ggplot(aes(auc, model)) +
  geom_col(width = 0.2) +
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
  
    
    
