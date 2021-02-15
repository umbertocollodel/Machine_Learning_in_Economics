# 1) Pre-processing of text data 
# 2) Method
# 3) Hyperparameter tuning
# 4) Length of training data

############ INSTRUCTIONS TO RUN ###############

# Change local_path to local directory where train and test.csv are lodged 
# Change intermediate_path to local directory where you want all cleaned rds to be stored

##################################################

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
# Three differens methods chosen: single words, bigrams and single words stemmed

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
  
  
# Compare stemming with single words tokenization (to see if actually common roots between words)
  
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
# Running three models on same trainining data with, however, text processed in different ways
# Unconditional mean common to all samples (same training data)

set.seed(436)


model <- label_train_x %>% 
  map2(label_train_y, ~ SuperLearner(Y= .y$Author, 
                      X= .x,
                      family = binomial(),
                      SL.library = c("SL.mean",
                                     "SL.kernelKnn",
                                     "SL.glmnet"),
                      cvControl = list(0)))


# Evaluate model on basis of already labelled "test" set performance -----


# Obtain fitted probabilities for labelled "test" set:

fitted <- model %>% 
    map2(label_test_x, ~ predict(.x, .y, onlySL = F)[["library.predict"]]) %>% 
    map(~ as_tibble(.x))


# Rank models by Area under the Curve:

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


# Plot performance:
  

auc_df %>%
  mutate(models = factor(models)) %>% 
  mutate(models = fct_reorder(models,auc, mean)) %>% 
  ggplot(aes(auc, models)) +
  geom_col(width = 0.2) +
  facet_wrap(~ pre_processing) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank()) +
  xlab("Area under the Curve (AUC)") +
  ylab("Algorithm") +
  theme(axis.text.x = element_text(angle = 270)) +
  theme(strip.text.x = element_text(size=14),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title = element_text(size = 22))


# Export:

export_path="../Machine_learning_for_economics_material/output/homework_1/figures/"
ggsave(paste0(export_path,"performance.pdf"))



# Extend on "real" test set ----


# What was the optimizing threshold on the previous test set?

new <- fitted[[3]]$SL.glmnet_All %>% 
  data.frame(fitted = .) %>% 
  cbind(label_test_y[[3]]) %>% 
  as_tibble() 


# Calculate optimal threshold:


calculate_threshold <- function(data, threshold){
  
  cbind(data, threshold) %>% 
  mutate(classification = case_when(fitted > threshold ~ 1,
                                    T~ 0)) %>% 
  mutate(type_1 = case_when(Author == 1 & classification == 0 ~ 1,
                            T ~ 0)) %>% 
  mutate(type_2 = case_when(Author == 0 & classification == 1 ~ 1,
                            T ~ 0)) %>% 
  summarise_at(vars(contains("type")),funs(mean(.)*100)) %>% 
  mutate(loss = type_1 + type_2) %>% 
  .$loss
  
  
  
  }
  
  
seq(0, 1, by = 0.025) %>% 
  map(~ calculate_threshold(new,.x)) %>% 
  bind_rows()

  

# We use only the best model i.e. LASSO with words tokenization


predict(model[[3]], tidy_tweets_topwords[[6]] %>% select(-id))$pred %>% 
  data.frame(fitted = .) %>% 
  mutate(fitted = round(fitted,2)) %>% 
  mutate(dummy = case_when(fitted >0.5 ~ "bernie",
                           T ~ "trump")) %>% 
  ggplot(aes(fitted, fill = dummy)) +
  geom_density(col = "white",alpha = 0.4) +
  xlab("") +
  ylab("") +
  labs(fill = "") +
  theme_minimal() +
  scale_fill_manual(values = c("#0000ff","#ff0000")) +
  theme(legend.position = "bottom", 
        legend.text = element_text(size=18)) +
  theme(axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title = element_text(size = 22))

ggsave(paste0(export_path,"fitted.pdf"))
  
# Footnote:

footnote=c("")








  
    
    
