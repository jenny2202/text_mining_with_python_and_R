# Load data: 
rm(list = ls())
library(tidyverse)
library(dplyr)
library(magrittr)
sms_raw <- read.csv("/Users/jennynguyen/Downloads/sms_spam.csv")


# Extract label (spam or ham) and convert to factor: 
label <- sms_raw$type %>% as.factor()

# Word cloud for spams: 

par(bg = "black") 
set.seed(1709)
library(wordcloud)

wordcloud(sms_raw %>% filter(type == "spam") %>% pull(text), 
          max.words = 100, 
          random.order = FALSE, 
          rot.per = 0.35, 
          font = 2,
          colors = brewer.pal(8, "Dark2"))


# Preapre data for modelling: 

library(tm)

sms_corpus <- sms_raw$text %>% 
  VectorSource() %>% 
  VCorpus() %>% 
  tm_map(content_transformer(tolower)) %>% # Convert the text to lower case
  tm_map(removeNumbers) %>% # Remove numbers
  tm_map(removeWords, stopwords()) %>% # Remove stop word like "we", "you"
  tm_map(removePunctuation) %>% # Remove punctuations
  tm_map(stripWhitespace) # Eliminate extra white spaces

# Convert to DTM sparse matrix: 

dtm <- sms_corpus %>% DocumentTermMatrix()

# List of words that appear more than 20: 

at_least20 <- findFreqTerms(dtm, 20)

# Convert sparse matrix to data frame: 

inputs <- apply(dtm[, at_least20], 2, 
                function (x) {case_when(x == 0 ~ "No", TRUE ~ "Yes") %>% as.factor()})

df = inputs %>% 
  as.matrix() %>% 
  as.data.frame()  %>% 
  mutate(Class = label)

# Split data: 

library(caret)
set.seed(1)
id <- createDataPartition(y = df$Class, p = 0.7, list = FALSE)
df_train_ml <- df[id, ]
df_test_ml <- df[-id, ]


# Use Parallel computing: 

library(doParallel)
registerDoParallel(cores = detectCores() - 1)

# Activate h2o package for using: 

library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16g")

# Convert to h2o Frame and identify inputs and output: 

h2o.no_progress()
test <- as.h2o(df_test_ml)
train <- as.h2o(df_train_ml)

y <- "Class"
x <- setdiff(names(train), y)


# Set hyperparameter grid: 

hyper_grid.h2o <- list(ntrees = seq(50, 500, by = 50),
                       mtries = seq(3, 5, by = 1),
                       # max_depth = seq(10, 30, by = 10),
                       # min_rows = seq(1, 3, by = 1),
                       # nbins = seq(20, 30, by = 10),
                       sample_rate = c(0.55, 0.632, 0.75))



# Set random grid search criteria: 

search_criteria_2 <- list(strategy = "RandomDiscrete",
                          stopping_metric = "AUC",
                          stopping_tolerance = 0.005,
                          stopping_rounds = 10,
                          max_runtime_secs = 30*60)


# Turn parameters for RF: 

system.time(random_grid <- h2o.grid(algorithm = "randomForest",
                                    grid_id = "rf_grid2",
                                    x = x, 
                                    y = y, 
                                    seed = 29, 
                                    nfolds = 10, 
                                    training_frame = train,
                                    hyper_params = hyper_grid.h2o,
                                    search_criteria = search_criteria_2))


# Collect the results and sort by our models: 
grid_perf2 <- h2o.getGrid(grid_id = "rf_grid2", 
                          sort_by = "AUC", 
                          decreasing = FALSE)

# Best RF: 
best_model2 <- h2o.getModel(grid_perf2@model_ids[[1]])


# ROC curve and AUC: 
library(pROC) 

# Function calculates AUC: 

auc_for_test <- function(model_selected) {
  actual <- df_test_ml$Class
  pred_prob <- h2o.predict(model_selected, test) %>% as.data.frame() %>% pull(spam)
  return(roc(actual, pred_prob))
}

# Use this function: 
my_auc <- auc_for_test(best_model2)

# Graph ROC and AUC: 

sen_spec_df <- data_frame(TPR = my_auc$sensitivities, FPR = 1 - my_auc$specificities)

theme_set(theme_minimal())

sen_spec_df %>% 
  ggplot(aes(x = FPR, ymin = 0, ymax = TPR))+
  geom_polygon(aes(y = TPR), fill = "red", alpha = 0.3)+
  geom_path(aes(y = TPR), col = "firebrick", size = 1.2) +
  geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
  theme_bw() +
  coord_equal() +
  labs(x = "FPR (1 - Specificity)", 
       y = "TPR (Sensitivity)", 
       title = "Model Performance for RF Classifier", 
       subtitle = paste0("AUC Value: ", my_auc$auc %>% round(2)))




# Set a range of threshold for classification: 

my_threshold <- c(0.10, 0.15, 0.35, 0.4, 0.45, 0.5)

# Use this best model for prediction with some thresholds selected: 

my_cm_com_rf_best2 <- function(thre) {
  
  du_bao_prob <- h2o.predict(best_model2, test) %>% as.data.frame() %>% pull(spam)
  du_bao <- case_when(du_bao_prob >= thre ~ "spam", du_bao_prob < thre ~ "ham") %>% as.factor()
  cm <- confusionMatrix(du_bao, df_test_ml$Class, positive = "spam")
  return(cm)
  
}

# Model Performance by cutoff selected: 

results_list_rf <- lapply(my_threshold, my_cm_com_rf_best2)

# Function for presenting prediction power by class:  

vis_detection_rate_rf <- function(x) {
  
  results_list_rf[[x]]$table %>% as.data.frame() -> m
  my_acc <- results_list_rf[[x]]$table %>% as.vector()
  k <- my_acc[4] / sum(my_acc[c(2, 4)])
  rate <- round((100*my_acc[4] / sum(my_acc[c(2, 4)])) , 2)
  acc <- round(100*sum(m$Freq[c(1, 4)]) / sum(m$Freq), 2)
  acc <- paste0(acc, "%")
  
  m %>% 
    ggplot(aes(Prediction, Freq, fill = Reference)) +
    geom_col(position = "fill") + 
    scale_fill_manual(values = c("#e41a1c", "#377eb8"), name = "") + 
    theme(panel.grid.minor.y = element_blank()) + 
    theme(panel.grid.minor.x = element_blank()) + 
    scale_y_continuous(labels = scales::percent) + 
    labs(x = NULL, y = NULL, 
         title = paste0("Model Performance when Threshold = ", my_threshold[x]), 
         subtitle = paste0("Accuracy for Spam Cases: ", rate, "%", ", ", "Accuracy: ", acc))
}

# Use this function: 

gridExtra::grid.arrange(vis_detection_rate_rf(1), 
                        vis_detection_rate_rf(2), 
                        vis_detection_rate_rf(3), 
                        vis_detection_rate_rf(4), 
                        vis_detection_rate_rf(5), 
                        vis_detection_rate_rf(6))




# Model performance when cutoff = 0.35: 
results_list_rf[[3]]$table
```