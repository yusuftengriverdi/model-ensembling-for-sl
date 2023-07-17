library(caret)
library(ggplot2)
library(FNN)
library(MLmetrics)

df_train <- read.csv("data/train_ch.csv", header = TRUE)
df_train <- df_train[, -c(1)]

preproc <- preProcess(df_train, method = c("center", "scale"))
normalized_train_df <- predict(preproc, df_train)
normalized_train_df <- as.data.frame(normalized_train_df)

df_train_mod <- normalized_train_df
df_train_mod$avg_v1_v5_v7 <- rowMeans(df_train_mod[, c("v1", "v5", "v7")])
df_train_mod <- df_train_mod[, -c(1, 5, 7)]
df_train_mod <- df_train_mod[, c(names(df_train_mod)[-length(df_train_mod)], names(df_train_mod)[length(df_train_mod)])]


df_train_mod_2 <- df_train_mod
df_train_mod_2$avg_v8_v9 <- rowMeans(df_train_mod_2[, c("v8", "v9")])
df_train_mod_2 <- df_train_mod_2[, -c(5, 6)]
df_train_mod_2 <- df_train_mod_2[, c(names(df_train_mod_2)[-length(df_train_mod_2)], names(df_train_mod_2)[length(df_train_mod_2)])]

regress_cols <- c("v3", "avg_v1_v5_v7", "avg_v8_v9")

X_reg_train <- df_train_mod_2[, regress_cols]
y_reg_train <- df_train_mod_2$Y

# regressor <- lm(y_reg_train ~ ., data = cbind(X_reg_train, y_reg_train))
# print(regressor$results)

knn_cols <- c("v3", "avg_v1_v5_v7", "avg_v8_v9")

X_knn_train <- df_train_mod_2[, knn_cols]
y_knn_train <- df_train_mod_2$Y

r2_scores <- numeric()
# Iterate over k values from 1 to 15
for (k in 1:15) {
  # Fit k-nearest neighbors model
  cluster = knn.reg(train=X_knn_train, y=y_knn_train, k = k, algorithm="kd_tree")
  
  # Store R2 score in the array
  r2_scores <- c(r2_scores, cluster$R2Pred)
}

# Plot the R2 scores
plot(1:15, r2_scores, type = "b", pch = 19, xlab = "k", ylab = "R2 Score",
     main = "R2 Score vs. k", xlim = c(1, 15), ylim = c(0, 1))

k = 10
cluster = knn.reg(train=X_knn_train, y=y_knn_train, k = k, algorithm="kd_tree")
print(cluster$R2Pred)
print(RMSE(cluster$pred, y_knn_train))

cv_metrics <- c("RMSE", "Rsquared")
cv_results <- train(X_reg_train, y_reg_train, method = "lm", trControl = trainControl(method = "repeatedcv", number = 15, repeats = 10))
print(cv_results$results)


df_test <- read.csv("data/test_ch.csv", header = TRUE)
df_test <- df_test[, -c(1)]

preproc <- preProcess(df_test, method = c("center", "scale"))
normalized_test_df <- predict(preproc, df_test)
normalized_test_df <- as.data.frame(normalized_test_df)

df_test_mod <- normalized_test_df
df_test_mod$avg_v1_v5_v7 <- rowMeans(df_test_mod[, c("v1", "v5", "v7")])
df_test_mod <- df_test_mod[, -c(1, 5, 7)]
df_test_mod <- df_test_mod[, c(names(df_test_mod)[-length(df_test_mod)], names(df_test_mod)[length(df_test_mod)])]


df_test_mod_2 <- df_test_mod
df_test_mod_2$avg_v8_v9 <- rowMeans(df_test_mod_2[, c("v8", "v9")])
df_test_mod_2 <- df_test_mod_2[, -c(5, 6)]
df_test_mod_2 <- df_test_mod_2[, c(names(df_test_mod_2)[-length(df_test_mod_2)], names(df_test_mod_2)[length(df_test_mod_2)])]

regress_cols <- c("v3", "avg_v1_v5_v7", "avg_v8_v9")

X_reg_test <- df_test_mod_2[, regress_cols]
y_reg_test <- df_test_mod_2$Y

# regressor <- lm(y_reg_test ~ ., data = cbind(X_reg_test, y_reg_test))
# print(regressor$results)

knn_cols <- c("v3", "avg_v1_v5_v7", "avg_v8_v9")

X_knn_test <- df_test_mod_2[, knn_cols]
y_knn_test <- df_test_mod_2$Y

cluster_test = knn.reg(train=X_knn_train, test=X_knn_test, y=y_knn_train, k = k, algorithm="kd_tree")

lm_predictions = predict(cv_results, newdata= X_knn_test)
# Create a data frame with the predicted values
predictions <- data.frame(pred_knn = cluster_test$pred, pred_lm = lm_predictions)

# Save the data frame as a CSV file
write.csv(predictions, "predictions.csv", row.names = FALSE)


y_reg_test_completo <- df_test_completo_mod_2$Y

lm_score = MSE(predictions$pred_lm, y_reg_test_completo)
knn_score = MSE(predictions$pred_knn, y_reg_test_completo)