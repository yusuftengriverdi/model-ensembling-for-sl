#Adapted from the caret vignette
pacman::p_load(caret, caretEnsemble, dplyr, mlbench, ggplot2, pROC, ROCR, mltools, readr, ModelMetrics, MLmetrics, ctv, car, ggplot2, install = T)

# Get Data on ADCTL.
ADCTLtrain <- read_csv("ADCTLtrain.csv")
ADCTLtest <- read_csv("ADCTLtest.csv")

# Process data. Whatever you do, apply to test data set too.
ADCTLtrain = as.data.frame(select(ADCTLtrain, -ID))
ADCTLtest = as.data.frame(select(ADCTLtest, -ID))

# If necessary, we can use X-y formula.
y = ADCTLtrain$Label
X = as.data.frame(select(ADCTLtrain, -Label))
# Preprocess pipeline.
preproc <- preProcess(ADCTLtrain, method = c("center", "scale"))

train_val_set <- predict(preproc, ADCTLtrain)
# We should apply same transformation to ADCTL test, because otherwise we will not get good result.
test_set <- predict(preproc, ADCTLtest)

# Feature reduction with PCA.
train_val.pca <- prcomp(scale(X), center=T)

pca_var <- train_val.pca$sdev^2
pca_var_perc <- round(pca_var/sum(pca_var) * 100, 1)
barplot(pca_var_perc, main = "Variation Plot", xlab = "PCs", ylab = "Percentage Variance", ylim = c(0, 45), xlim=c(0, 50))

PC1 <- train_val.pca$rotation[,1]
PC1_scores <- abs(PC1)
PC1_scores_ordered <- sort(PC1_scores, decreasing = TRUE)

pca_threshold = 0.02
selected_features <- Filter(function(x) x > pca_threshold, PC1_scores_ordered)
# names(PC1_scores_ordered)

# ggplot(train_val_set, aes(x=TOMM7, y=ATP5O, color = Label)) + geom_point() + labs(title = 'The two most impactful features')
reduced_train_val_set <- subset(train_val_set, select=names(selected_features))
reduced_train_val_set$Label <- train_val_set$Label

reduced_test_set <- subset(test_set, select=names(selected_features))

# Make a named, ordered factor target column - because that's what works with all models.
train_val_set_factor <- reduced_train_val_set 
y_ = ordered(reduced_train_val_set$Label, levels = c("AD", "CTL"), labels = c(0, 1))
y_ <- make.names(levels(y_))[y_]
train_val_set_factor$Label <- y_
colnames(train_val_set_factor) <- make.names(colnames(train_val_set_factor))

# Set the random seed for reproducibility
set.seed(123)
# Calculate the sample size.
# Perform random sampling with class distribution preservation.
validation_indices <- caret::createDataPartition(train_val_set_factor$Label, times = 1, p = 0.35, list = TRUE,
                                                 groups = min(5, length(train_val_set_factor$Label)))
# Extract the sampled validation data set.
val_set <- train_val_set_factor[validation_indices$Resample1, ]
train_set <- train_val_set_factor[-validation_indices$Resample1, ]

# This is our custom summary function, to see scores in a table.
# This is our custom summary function, to see scores in a table.
customSummary <- function(data, lev = NULL, model = NULL) {
  
  auc <- mltools::auc_roc(data$pred, factor(data$obs, ordered=TRUE))
  mcc <- mltools::mcc(data$pred, data$obs)
  rmse <- mltools::rmse(as.numeric(data$pred), as.numeric(data$obs))
  mslerr <- msle(data$pred, data$obs)
  acc <- Accuracy(data$pred, data$obs)
  r2score <- R2_Score(as.numeric(data$pred),as.numeric(data$obs))
  f1 <- MLmetrics::F1_Score(as.numeric(data$pred),as.numeric(data$obs))
  sens <- MLmetrics::Sensitivity(as.numeric(data$pred),as.numeric(data$obs))
  spec <- MLmetrics::Specificity(as.numeric(data$pred),as.numeric(data$obs))
  rec <- MLmetrics::Recall(as.numeric(data$pred),as.numeric(data$obs))
  poissonlog <- MLmetrics::Poisson_LogLoss(as.numeric(data$pred),as.numeric(data$obs))
  # Return a list with AUC and ROC curve data
  out = c(auc,mcc, rmse, mslerr, acc, r2score, f1, sens, spec, rec, poissonlog)
  names(out) <- c("AUC", "MatthewsCorr", "RMSE", "MSLE", "Accuracy", "R2Score", "F1Score", "Sensitivity", "Specifity", "Recall", "PoissonLogLoss")
  out
}


set.seed(107)

training <- train_set
validation <- val_set
testing <- reduced_test_set
## TRY WITH REPEATEDCV
my_control <- trainControl(
  method="repeatedcv",
  number=5,
  repeats=5,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(training$Label, 25),
  summaryFunction=customSummary
)

model_list <- caretList(
  Label ~ ., 
  data=training,
  trControl=my_control,
  methodList=c(
    "bayesglm",
    "mlp", 
    "fda", 
    "gaussprLinear", 
    "regLogistic", 
    "glmnet", 
    "svmRadial", 
    "knn", 
    "svmPoly",
    "rf",
    "gbm",
    "pls",
    "svmLinear",
    "gaussprRadial",
    "xgbLinear"
  )
)
get_results <- function(model_list) {
  
  # Create an empty data frame to store the results.
  results <- data.frame(Model = character(),
                        AUC = numeric(), 
                        MatthewsCorr = numeric(), 
                        RMSE = numeric(), 
                        MSLE = numeric(), 
                        Accuracy=numeric(),
                        R2Score = numeric(),
                        stringsAsFactors = FALSE)
  
  # Add results to the data frame.
  for (i in seq_along(model_list)) {
    results[nrow(results) + 1, ] <- list(Model = model_list[[i]]$modelInfo$label,
                                         AUC = mean(model_list[[i]]$results$AUC), 
                                         MatthewsCorr = mean(model_list[[i]]$results$MatthewsCorr), 
                                         RMSE = mean(model_list[[i]]$results$RMSE), 
                                         MSLE = mean(model_list[[i]]$results$MSLE), 
                                         Accuracy = mean(model_list[[i]]$results$Accuracy),
                                         R2Score = mean(model_list[[i]]$results$R2Score),
                                         F1Score = mean(model_list[[i]]$results$F1Score),
                                         Sensitivity = mean(model_list[[i]]$results$Sensitivity),
                                         Specificity = mean(model_list[[i]]$results$Specificity),
                                         Recall = mean(model_list[[i]]$results$Recall),
                                         PoissonLogLoss = mean(model_list[[i]]$results$PoissonLogLoss)
                                         )
    
  }
  results  
}

train_results <- get_results(model_list = model_list)

ens_control <- trainControl(
  method="boot",
  number=5,
  savePredictions="final",
  classProbs=TRUE,
  summaryFunction=customSummary
)

ensembled_v3 <- caretEnsemble(model_list, metric='AUC', trControl= ens_control)

val_preds = predict(ensembled_v3, newdata = val_set)


val_obs = ordered((val_set$Label), levels = c("X0", "X1"), labels = c(0, 1))
val_obs <- make.names(levels(val_obs))[val_obs]
val_obs <- as.factor(val_obs)

# Bagging with Decorrelated Predictors
baggedPredictions <- lapply(model_list, predict, newdata = val_set)

model_preds <- baggedPredictions
model_preds$caretEnsemble <- val_preds

# Stacked Generalization with Cross-Validation
stackedData <- baggedPredictions
names(stackedData) <- names(model_list)
stackedData$Label = val_obs

meta_control <- trainControl(
  method="boot",
  number=5,
  savePredictions="final",
  classProbs=TRUE,
  summaryFunction=customSummary
)

# TRY WITH ANOTHER MODEL
metaModel <- train(Label ~ ., data = as.data.frame(stackedData), method = "rf", trControl=meta_control)

# model_preds$meta <- metaModel$finalModel$predicted

get_val_results <- function(model_preds) {
  # Create an empty data frame to store the results.
  results <- data.frame(Model = character(),
                        AUC = numeric(), 
                        MatthewsCorr = numeric(), 
                        RMSE = numeric(), 
                        MSLE = numeric(), 
                        Accuracy=numeric(),
                        R2Score = numeric(),
                        stringsAsFactors = FALSE)
  
  # Add results to the data frame.
  for (i in seq_along(model_preds)) {
    summary = customSummary(list(preds=model_preds[[i]], obs=val_obs))
    results[nrow(results) + 1, ] <- list(Model = names(model_preds)[[i]],
                                         AUC = summary["AUC"], 
                                         MatthewsCorr = summary["MatthewsCorr"], 
                                         RMSE = summary["RMSE"], 
                                         MSLE = summary["MSLE"], 
                                         Accuracy = summary["Accuracy"],
                                         R2Score = summary["R2Score"],
                                         F1Score = summary["F1Score"],
                                         Sensitivity = summary["Sensitivity"],
                                         Specificity = summary["Specifity"],
                                         Recall = summary["Recall"],
                                         PoissonLogLos = summary["PoissonLogLoss"]
                                         
    )
    
  }
  results  
}

## Use caretStack with Gradient Linear Model.
glm_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="AUC",
  trControl=trainControl(
    method="boot",
    number=5,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=customSummary
  )
)
model_preds$ensemble <- predict(glm_ensemble, newdata=val_set)


## Use caretStack with Random Forest.
rf_ensemble <- caretStack(
  model_list,
  method="rf",
  metric="AUC",
  trControl=trainControl(
    method="boot",
    number=5,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=customSummary
  )
)
model_preds$rf_ensemble <- predict(rf_ensemble, newdata=val_set)


# Define the weights for each model
model_weights <- list(glmnet=mean(model_list$glmnet$results$AUC),
                      svmRadial= mean(model_list$svmRadial$results$AUC), 
                      rf = mean(model_list$rf$results$AUC), 
                      gbm = mean(model_list$gbm$results$AUC),
                      pls = mean(model_list$pls$results$AUC),
                      svmLinear = mean(model_list$svmLinear$results$AUC), 
                      gaussprRadial = mean(model_list$gaussprRadial$results$AUC), 
                      xgbLinear = mean(model_list$xgbLinear$results$AUC),
                      svmPoly = mean(model_list$svmPoly$results$AUC),
                      knn = mean(model_list$knn$results$AUC))

# Calculate the sum of weights
weight_sum <- sum(as.numeric(model_weights))

weightedPredictions <- baggedPredictions

for (i in seq(10)) {
  weightedPredictions[[i]] <- (model_weights[[i]] * as.numeric(baggedPredictions[[i]])) / weight_sum
}

baggedEnsemblePred <- rowMeans(do.call(cbind, weightedPredictions))
bagged_result <- as.data.frame(customSummary(list(preds=baggedEnsemblePred, obs=as.numeric(val_obs))))


val_results <- get_val_results(model_preds)
model_preds$bagged <- baggedEnsemblePred
val_results$bagged <- bagged_result


# This is our custom summary function, to see scores in a table.
customProbSummary <- function(data, lev = NULL, model = NULL) {
  
  auc <- mltools::auc_roc(data$pred, data$obs)
  mcc <- mltools::mcc(data$pred, data$obs)
  rmse <- mltools::rmse(as.numeric(data$pred), data$obs)
  mslerr <- msle(data$pred, data$obs)
  acc <- Accuracy(data$pred, data$obs)
  r2score <- R2_Score(data$pred,data$obs)
  f1 <- ModelMetrics::f1Score(data$pred,data$obs)
  sens <- ModelMetrics::sensitivity(data$pred,data$obs)
  spec <- ModelMetrics::specificity(data$pred,data$obs)
  rec <- ModelMetrics::recall(data$pred,data$obs)
  kappa <- ModelMetrics::kappa(data$pred,data$obs)
  # Return a list with AUC and ROC curve data
  out = c(auc,mcc, rmse, mslerr, acc, r2score, f1, sens, spec, rec, kappa)
  names(out) <- c("AUC", "MatthewsCorr", "RMSE", "MSLE", "Accuracy", "R2Score", 
                  "F1 Score", "Sensitivity", "Specifity", "Recall", "Kappa")
  out
}

model_probs = list()
# Add results to the data frame.
for (i in seq_along(model_list)) {

  probs <- list(Model = model_list[[i]]$modelInfo$label,
                train_probX0 = predict(model_list[[i]], newdata = train_set, type = "prob")$X0,
                train_probX1 = predict(model_list[[i]], newdata = train_set, type = "prob")$X1,
                train_pred = predict(model_list[[i]], newdata = train_set),
                val_probX0 = predict(model_list[[i]], newdata = val_set, type = "prob")$X0,
                val_probX1 = predict(model_list[[i]], newdata = val_set, type = "prob")$X1,
                val_pred = predict(model_list[[i]], newdata = val_set)
  )
  model_probs[[i]] <- probs
  
}

names(model_probs) <- names(model_list)


# Add results to the data frame.
for (i in seq_along(model_probs)) {
  train_prob <-  Map(c, model_probs[[i]]$train_probX0, model_probs[[i]]$train_probX1 )
  val_prob <-  Map(c, model_probs[[i]]$val_probX0, model_probs[[i]]$val_probX1 )
  train_pred <- model_probs[[i]]$train_pred
  val_pred <- model_probs[[i]]$val_pred
  
  tmp = as.data.frame(customProbSummary(list(pred=as.matrix(train_prob), obs=model.matrix(~ train_set$Label - 1))))
}

# Create an empty plot
plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate")

# Loop over the models in the list
for (i in seq_along(model_list)) {
  # Compute ROC curve for each model
  roc_obj <- pROC::roc(response=train_set$Label, predictor=ordered(predict(model_list[[i]], newdata = train_set)))
  
  # Plot ROC curve
  plot(roc_obj, add = TRUE, col = i, print.auc = T, print.auc.x = 0.5, print.auc.y = 0.2 + 0.05 * i, legacy.axes = TRUE)
  
  # Add legend
  legend("bottomright", legend = names(model_list), col = 1:length(model_list), lty = 1:length(model_list), cex = 0.4)
}

# Let's try with a reduced list of ML classifiers.
model_list_wo <- caretList(
  Label ~ ., 
  data=train_val_set_factor,
  trControl=my_control,
  methodList=c(
    "bayesglm",
    "mlp", 
    "fda", 
    "gaussprLinear", 
    "regLogistic", 
    "glmnet", 
    "svmRadial", 
    "knn", 
    "svmPoly",
    "rf",
    "gbm",
    "pls",
    "svmLinear",
    "gaussprRadial",
    "xgbLinear"
  )
)

rf_ensemble_wo <- caretStack(
  model_list_wo,
  method="rf",
  metric="AUC",
  trControl=trainControl(
    method="boot",
    number=5,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=customSummary
  )
)

colnames(reduced_test_set) <- make.names(colnames(reduced_test_set))

# Bagging with Decorrelated Predictors
baggedPredictions_wo <- lapply(model_list_wo, predict, newdata = reduced_test_set)

submit_preds <- predict(rf_ensemble_wo$ens_model, newdata=reduced_test_set)
submit_probs <- predict(rf_ensemble_wo$ens_model, newdata=baggedPredictions_wo, type='prob')

ADCTLtest <- read_csv("ADCTLtest.csv")

list(ID=ADCTLtest$ID, preds=submit_preds, probs=)
write.csv(submit_preds, file = "preds.csv", row.names = FALSE)
write.csv(submit_probs, file = "probs.csv", row.names = FALSE)
