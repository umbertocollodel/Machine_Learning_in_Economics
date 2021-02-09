
library(tidyverse)
library(stopwords)
library(tidytext)

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

list_df %>% 
  map(~ .x %>% mutate(Tweet = str_replace_all(Tweet, replace_reg, ""))) %>% 
  map(~ .x %>% unnest_tokens(word, Tweet, "regex",pattern = unnest_reg, to_lower = T)) %>% 
  map(~ .x %>% filter(!word %in% stpwds & str_detect(word, "[a-z]"))) %>% 
  