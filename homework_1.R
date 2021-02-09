
library(tidyverse)
library(stopwords)
library(tidytext)
library(reshape2)
library(SuperLearner)
library(tidymodels)

name_files=c("test","train")


list_df <- name_files %>% 
  map_chr(~ paste0("../Machine_learning_for_economics_material/raw_data/homework_1/",.x,".csv")) %>% 
  map(~ read.csv(.x)) %>% 
  map(~ as_tibble(.x))

names(list_df) <- name_files



# 

stpwds = stopwords(source = "stopwords-iso") 
 
replace_reg <- "http[s]?://[A-Za-z\\d/\\.]+|&amp;|&lt;|&gt;"
unnest_reg  <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"


# why only words and not other n-grams?

clean_list_df <- list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "regex",pattern = unnest_reg, to_lower = T)) %>% 
  map(~ .x %>% filter(!word %in% stpwds & str_detect(word, "[a-z]"))) 


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
    
tidy_tweets_topwords_train <- tidy_tweets_train %>% 
      mutate(topwords = ifelse(word %in% freq_words$word, 1,0)) %>%
      mutate(word = ifelse(topwords==1, word, "no_top_word")) %>%
      unique() %>%
      group_by(id) %>%
      mutate(notopwords = 1-max(topwords)) %>%
      ungroup() %>%
      filter(!(word=="no_top_word" & notopwords==0)) %>%
      select(-topwords, -notopwords) %>%
      unique() %>%
      reshape2::dcast(id+Author~word, function(x) 1, fill = 0)
    

    
  tidy_tweets_topwords_test <- tidy_tweets_test %>% 
      mutate(topwords = ifelse(word %in% freq_words$word, 1,0)) %>%
      mutate(word = ifelse(topwords==1, word, "no_top_word")) %>%
      unique() %>%
      group_by(id) %>%
      mutate(notopwords = 1-max(topwords)) %>%
      ungroup() %>%
      filter(!(word=="no_top_word" & notopwords==0)) %>%
      select(-topwords, -notopwords) %>%
      unique() %>%
      reshape2::dcast(id~word, function(x) 1, fill = 0) 
    

    
  # Save intermediate files ----
    
    list(tidy_tweets_topwords_train,tidy_tweets_topwords_test) %>% 
      walk2(name_files, ~ saveRDS(.x, file = paste0("../Machine_learning_for_economics_material/intermediate_data/homework_1/",.y,"_top500_words.rds")))
 
  
  
  # Train the model ----
  
  set.seed(243)
  
  a <- SuperLearner(Y= ifelse(tidy_tweets_topwords_train$Author == "bernie",1,0), 
               X= tidy_tweets_topwords_train %>% select(-id,-Author),
               family = binomial(),
               SL.library = c("SL.mean",
                              "SL.kernelKnn",
                              "SL.randomForest"))

  predicted <- predict(a, tidy_tweets_topwords_test %>% select(-id)) 

  data.frame(fitted_knn = predicted$library.predict$SL.kernelKnn_All) %>% 
  ggplot(aes(x=fitted)) +
    geom_density() +
    theme_minimal()
    
    
    
    
