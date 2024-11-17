library(e1071) 
library(rpart)  
library(ROCR)   
library(rpart.plot)
library(knitr)  

# Loading the dataset
data <- read.csv("File Path")

# Converting  categorical variables to factors
data$GENDER <- as.factor(data$GENDER)
data$LUNG_CANCER <- as.factor(data$LUNG_CANCER)

# Preprocessing the data
numeric_cols <- sapply(data, is.numeric)
preprocessed_data <- scale(data[, numeric_cols])

# Performing PCA
pca_result <- prcomp(preprocessed_data)

# Splitting the dataset into training and testing sets
set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(data), 0.8 * nrow(data))  # 80% training data
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Naive Bayes classifier
naive_bayes_model <- naiveBayes(LUNG_CANCER ~ ., data = train_data)

# Predicting
naive_bayes_pred <- predict(naive_bayes_model, newdata = test_data)

# Evaluating Naive Bayes model
confusion_matrix_nb <- table(naive_bayes_pred, test_data$LUNG_CANCER)
accuracy_nb <- sum(diag(confusion_matrix_nb)) / sum(confusion_matrix_nb)

# Predicting probabilities for Naive Bayes
naive_bayes_probs <- predict(naive_bayes_model, newdata = test_data, type = "raw")

# Create prediction object 
naive_bayes_pred_obj <- prediction(naive_bayes_probs[, "YES"], test_data$LUNG_CANCER)

# Calculate  the performance for Naive Bayes
naive_bayes_perf <- performance(naive_bayes_pred_obj, "tpr", "fpr")

# Calculate AUC for Naive Bayes
naive_bayes_auc <- performance(naive_bayes_pred_obj, "auc")
auc_naive_bayes <- as.numeric(naive_bayes_auc@y.values)

# Train the decision tree model
dt_model <- rpart(LUNG_CANCER ~ ., data = train_data)

#  predictions using the decision tree model
dt_predictions_prob <- predict(dt_model, newdata = test_data, type = "prob")

# Create prediction object for Decision Tree
dt_pred_obj <- prediction(dt_predictions_prob[, 2], test_data$LUNG_CANCER)

# Calculate performance for decision tree
dt_perf <- performance(dt_pred_obj, "tpr", "fpr")

# Calculate AUC for decision tree
dt_auc <- performance(dt_pred_obj, "auc")
auc_dt <- as.numeric(dt_auc@y.values)

# Create a summary table
summary_table <- data.frame(
  Model = c("Naive Bayes", "Decision Tree"),
  Accuracy = c(accuracy_nb, confusion_matrix_nb[2, 2]/sum(confusion_matrix_nb[2,])),
  AUC = c(auc_naive_bayes, auc_dt)
)

# Print the summary table
print(summary_table)

# Plot ROC curve for Naive Bayes
plot(naive_bayes_perf, col = "blue", main = "ROC Curve for Naive Bayes", lwd = 2, 
     xlab = "False Positive Rate", ylab = "True Positive Rate")

# Plot ROC curve for Decision Tree
plot(dt_perf, col = "red", main = "ROC Curve for Decision Tree", lwd = 2, 
     xlab = "False Positive Rate", ylab = "True Positive Rate")

# Plot the decision tree
rpart.plot(dt_model, yesno = 2, type = 4, extra = 1, fallen.leaves = TRUE)

