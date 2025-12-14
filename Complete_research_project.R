# BA 6933 RESEARCH PROJECT - COMPLETE CODE
# Customer Lifetime Value Prediction and Churn Analysis with Regression Methods
# 
# Author:             Sai ram Podili
# Institution:        Trine University
# Course:             Statistics & Quantitative Methods (BA-6933-5O2--OL-FA-2025)
# Instructor:         Dr. Shane Allua
# Submission Date:    13th December 2025
#
# This file contains all code for Phases 2-4 of the research project:
# - Phase 2: Data Preprocessing & Exploratory Data Analysis
# - Phase 3: Model Development (Simple, Multiple, Logistic Regression)
# - Phase 4: Model Evaluation & Optimization
################################################################################

# ==============================================================================
# SETUP AND INITIALIZATION
# ==============================================================================

# Clear workspace
rm(list = ls())

# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(corrplot)     # Correlation plots
library(gridExtra)    # Arrange multiple plots
library(scales)       # For percentage formatting
library(caret)        # For confusion matrix and model evaluation
library(pROC)         # For ROC curve analysis
library(car)          # For VIF (multicollinearity check)
library(MASS)         # For stepwise regression
library(lmtest)       # For statistical tests

# Set seed for reproducibility
set.seed(123)

# Set working directory (ADJUST THIS TO YOUR PATH)
# setwd("your/path/here")

# Create output directories
dir.create("plots", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

cat("================================================================================\n")
cat("BA 6933 RESEARCH PROJECT: CLV PREDICTION AND CHURN ANALYSIS\n")
cat("================================================================================\n\n")

################################################################################
# PHASE 2: DATA PREPROCESSING & EXPLORATORY DATA ANALYSIS
################################################################################

# ------------------------------------------------------------------------------
# 2.1 DATA LOADING AND INITIAL INSPECTION
# ------------------------------------------------------------------------------

cat("--- 2.1 Data Loading ---\n")

# Load the Telco Customer Churn dataset
telco <- read.csv("~/Desktop/Project/WA_Fn-UseC_-Telco-Customer-Churn.csv", 
                  stringsAsFactors = FALSE)

cat(sprintf("Dataset loaded: %d rows, %d columns\n", nrow(telco), ncol(telco)))
print(colnames(telco))    # Column Names:
print(head(telco, 3))     # First few rows:

# ------------------------------------------------------------------------------
# 2.2 DATA CLEANING
# ------------------------------------------------------------------------------

cat("\n--- 2.2 Data Cleaning ---\n")

# Check for missing values
missing_summary <- colSums(is.na(telco))
print(missing_summary[missing_summary > 0])   # Missing values by column:

# Convert TotalCharges to numeric (handles blank spaces)
telco$TotalCharges <- as.numeric(telco$TotalCharges)

# Count missing TotalCharges
missing_charges <- sum(is.na(telco$TotalCharges))
cat(sprintf("\nRows with missing TotalCharges: %d (%.2f%%)\n", 
            missing_charges, 100*missing_charges/nrow(telco)))

# For customers with 0 tenure, set TotalCharges to 0
telco$TotalCharges[is.na(telco$TotalCharges) & telco$tenure == 0] <- 0

# Remove remaining rows with missing TotalCharges
telco_clean <- telco[!is.na(telco$TotalCharges), ]
cat(sprintf("Rows after cleaning: %d\n", nrow(telco_clean)))

# Create binary churn indicator
telco_clean$ChurnBinary <- ifelse(telco_clean$Churn == "Yes", 1, 0)

# ------------------------------------------------------------------------------
# 2.3 FEATURE ENGINEERING
# ------------------------------------------------------------------------------

cat("\n--- 2.3 Feature Engineering ---\n")

# Service Adoption Score: Count of services used
service_cols <- c("PhoneService", "MultipleLines", "InternetService", 
                  "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies")

telco_clean$ServiceAdoptionScore <- rowSums(
  sapply(telco_clean[, service_cols], function(x) x == "Yes")
)

# Customer Stability Index: Contract type weighted by tenure
telco_clean$StabilityIndex <- case_when(
  telco_clean$Contract == "Month-to-month" ~ telco_clean$tenure * 1,
  telco_clean$Contract == "One year" ~ telco_clean$tenure * 2,
  telco_clean$Contract == "Two year" ~ telco_clean$tenure * 3
)

# Revenue Capacity Score
telco_clean$RevenueCapacity <- telco_clean$MonthlyCharges * 
  telco_clean$ServiceAdoptionScore

# CLV (using TotalCharges as proxy)
telco_clean$CLV <- telco_clean$TotalCharges

#Feature engineering complete. New variables created:
#- ServiceAdoptionScore
#- StabilityIndex
#- RevenueCapacity
#- CLV
# ------------------------------------------------------------------------------
# 2.4 DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------

# Overall churn rate
churn_rate <- mean(telco_clean$ChurnBinary)
cat(sprintf("\nOverall Churn Rate: %.2f%%\n", churn_rate * 100))

# Key variables summary
key_vars <- c("tenure", "MonthlyCharges", "TotalCharges", "CLV", 
              "ServiceAdoptionScore", "StabilityIndex", "RevenueCapacity")
print(summary(telco_clean[, key_vars]))

# Correlation matrix
numeric_vars <- telco_clean[, c("tenure", "MonthlyCharges", "TotalCharges", 
                                "CLV", "ServiceAdoptionScore", 
                                "StabilityIndex", "RevenueCapacity",
                                "ChurnBinary")]
cor_matrix <- cor(numeric_vars, use = "complete.obs")
print(round(cor_matrix, 3))

# ------------------------------------------------------------------------------
# 2.5 VISUALIZATIONS
# ------------------------------------------------------------------------------

# 1. CLV Distribution
png("plots/01_clv_distribution.png", width = 800, height = 600)
hist(telco_clean$CLV, breaks = 50,
     main = "Distribution of Customer Lifetime Value (CLV)",
     xlab = "CLV (Total Charges)", col = "steelblue", border = "white")
abline(v = mean(telco_clean$CLV), col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("Mean CLV: $", round(mean(telco_clean$CLV), 2)),
       col = "red", lty = 2, lwd = 2)
dev.off()

# 2. Churn Distribution
png("plots/02_churn_distribution.png", width = 800, height = 600)
churn_counts <- table(telco_clean$Churn)
barplot(churn_counts, main = "Customer Churn Distribution",
        xlab = "Churn Status", ylab = "Count",
        col = c("green", "red"), ylim = c(0, max(churn_counts) * 1.2))
text(x = c(0.7, 1.9), y = churn_counts + 200, 
     labels = paste0(churn_counts, " (", 
                     round(100*churn_counts/sum(churn_counts), 1), "%)"))
dev.off()

# 3. CLV vs Tenure
png("plots/03_clv_vs_tenure.png", width = 800, height = 600)
plot(telco_clean$tenure, telco_clean$CLV,
     main = "CLV vs Tenure (Simple Linear Regression)",
     xlab = "Tenure (months)", ylab = "CLV (Total Charges)",
     col = alpha("steelblue", 0.5), pch = 16)
abline(lm(CLV ~ tenure, data = telco_clean), col = "red", lwd = 2)
cor_val <- cor(telco_clean$tenure, telco_clean$CLV)
legend("topleft", legend = paste("Correlation:", round(cor_val, 3)),
       bty = "n", cex = 1.2)
dev.off()

# 4. CLV vs Monthly Charges
png("plots/04_clv_vs_monthly_charges.png", width = 800, height = 600)
plot(telco_clean$MonthlyCharges, telco_clean$CLV,
     main = "CLV vs Monthly Charges",
     xlab = "Monthly Charges ($)", ylab = "CLV (Total Charges)",
     col = alpha("darkgreen", 0.5), pch = 16)
abline(lm(CLV ~ MonthlyCharges, data = telco_clean), col = "red", lwd = 2)
cor_val <- cor(telco_clean$MonthlyCharges, telco_clean$CLV)
legend("topleft", legend = paste("Correlation:", round(cor_val, 3)),
       bty = "n", cex = 1.2)
dev.off()

# 5. Correlation Heatmap
png("plots/05_correlation_heatmap.png", width = 1000, height = 900)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, addCoef.col = "black",
         number.cex = 0.7, title = "Correlation Matrix: Key Variables",
         mar = c(0, 0, 2, 0))
dev.off()

# 6. Churn by Contract Type
png("plots/06_churn_by_contract.png", width = 800, height = 600)
churn_by_contract <- prop.table(table(telco_clean$Contract, 
                                      telco_clean$Churn), 1) * 100
barplot(churn_by_contract[, "Yes"],
        main = "Churn Rate by Contract Type",
        xlab = "Contract Type", ylab = "Churn Rate (%)",
        col = "coral", ylim = c(0, 50))
text(x = c(0.7, 1.9, 3.1), y = churn_by_contract[, "Yes"] + 2,
     labels = paste0(round(churn_by_contract[, "Yes"], 1), "%"))
dev.off()

# 7. Churn by Tenure Groups
png("plots/07_churn_by_tenure.png", width = 800, height = 600)
telco_clean$TenureGroup <- cut(telco_clean$tenure, 
                               breaks = c(0, 12, 24, 48, 72),
                               labels = c("0-12", "13-24", "25-48", "49-72"))
churn_by_tenure <- prop.table(table(telco_clean$TenureGroup, 
                                    telco_clean$Churn), 1) * 100
barplot(churn_by_tenure[, "Yes"],
        main = "Churn Rate by Tenure Group",
        xlab = "Tenure (months)", ylab = "Churn Rate (%)",
        col = "skyblue", ylim = c(0, 60))
text(x = c(0.7, 1.9, 3.1, 4.3), y = churn_by_tenure[, "Yes"] + 2,
     labels = paste0(round(churn_by_tenure[, "Yes"], 1), "%"))
dev.off()

# 8. Service Adoption vs CLV
png("plots/08_service_adoption_vs_clv.png", width = 800, height = 600)
boxplot(CLV ~ ServiceAdoptionScore, data = telco_clean,
        main = "CLV by Service Adoption Score",
        xlab = "Number of Services Adopted", ylab = "CLV (Total Charges)",
        col = "lightblue", border = "navy")
dev.off()

# Save cleaned dataset
write.csv(telco_clean, "telco_customer_churn_cleaned.csv", row.names = FALSE)

# Phase 2 Complete
# Cleaned dataset saved: telco_customer_churn_cleaned.csv
# Visualizations saved in plots/ directory

################################################################################
# PHASE 3: MODEL DEVELOPMENT
################################################################################

# 3.1 SIMPLE LINEAR REGRESSION (CLV ~ Tenure)

simple_model <- lm(CLV ~ tenure, data = telco_clean)

# Regression Equation: 
coeffs <- coef(simple_model)
cat(sprintf("CLV = %.2f + %.2f × Tenure\n", coeffs[1], coeffs[2]))

# Model Summary:
summary_simple <- summary(simple_model)
print(summary_simple)

# Key Performance Metrics:
cat(sprintf("R-squared: %.4f\n", summary_simple$r.squared))
cat(sprintf("Adjusted R-squared: %.4f\n", summary_simple$adj.r.squared))
cat(sprintf("RMSE: $%.2f\n", sqrt(mean(simple_model$residuals^2))))

# Coefficient Interpretation:
cat(sprintf("Intercept: $%.2f - Expected CLV for new customer\n", coeffs[1]))
cat(sprintf("Tenure: $%.2f - Each additional month increases CLV by $%.2f\n", 
            coeffs[2], coeffs[2]))

# Visualization
png("plots/09_simple_regression_fit.png", width = 1000, height = 600)
par(mfrow = c(1, 2))
plot(telco_clean$tenure, telco_clean$CLV,
     main = "Simple Linear Regression: CLV ~ Tenure",
     xlab = "Tenure (months)", ylab = "CLV ($)",
     col = alpha("steelblue", 0.5), pch = 16, cex = 0.8)
abline(simple_model, col = "red", lwd = 3)
legend("topleft", 
       legend = c(sprintf("R² = %.3f", summary_simple$r.squared),
                  sprintf("CLV = %.2f + %.2f × Tenure", coeffs[1], coeffs[2])),
       bty = "n", cex = 1.1)

plot(simple_model$fitted.values, simple_model$residuals,
     main = "Residual Plot", xlab = "Fitted Values", ylab = "Residuals",
     col = alpha("darkgreen", 0.5), pch = 16)
abline(h = 0, col = "red", lwd = 2, lty = 2)
dev.off()

# ------------------------------------------------------------------------------
# 3.2 MULTIPLE LINEAR REGRESSION
# ------------------------------------------------------------------------------

multiple_model <- lm(CLV ~ tenure + MonthlyCharges + ServiceAdoptionScore, 
                     data = telco_clean)

# Regression Equation:
coeffs_mult <- coef(multiple_model)
cat(sprintf("CLV = %.2f + %.2f × Tenure + %.2f × MonthlyCharges + %.2f × ServiceAdoptionScore\n",
            coeffs_mult[1], coeffs_mult[2], coeffs_mult[3], coeffs_mult[4]))

# Model Summary:
summary_multiple <- summary(multiple_model)
print(summary_multiple)

# Key Performance Metrics:
cat(sprintf("R-squared: %.4f\n", summary_multiple$r.squared))
cat(sprintf("Adjusted R-squared: %.4f\n", summary_multiple$adj.r.squared))
cat(sprintf("RMSE: $%.2f\n", sqrt(mean(multiple_model$residuals^2))))

# Multicollinearity Check (VIF): 
vif_values <- vif(multiple_model)
print(vif_values)        # Note: VIF < 5 indicates acceptable multicollinearity

# Coefficient Interpretations:
cat(sprintf("Tenure: $%.2f per month (holding others constant)\n", coeffs_mult[2]))
cat(sprintf("MonthlyCharges: $%.2f per $1 (holding others constant)\n", coeffs_mult[3]))
cat(sprintf("ServiceAdoption: $%.2f per service (holding others constant)\n", coeffs_mult[4]))

# Model Comparison:
cat(sprintf("Simple Regression R²: %.4f\n", summary_simple$r.squared))
cat(sprintf("Multiple Regression R²: %.4f\n", summary_multiple$r.squared))
cat(sprintf("R² Improvement: %.4f (%.2f%% better)\n", 
            summary_multiple$r.squared - summary_simple$r.squared,
            100 * (summary_multiple$r.squared - summary_simple$r.squared) / summary_simple$r.squared))

# Diagnostic plots
png("plots/10_multiple_regression_diagnostics.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
plot(multiple_model)
dev.off()

# ------------------------------------------------------------------------------
# 3.3 LOGISTIC REGRESSION (Churn Prediction)
# ------------------------------------------------------------------------------

# Create dummy variables
telco_clean$Contract_TwoYear <- ifelse(telco_clean$Contract == "Two year", 1, 0)
telco_clean$Contract_OneYear <- ifelse(telco_clean$Contract == "One year", 1, 0)
telco_clean$InternetService_Fiber <- ifelse(telco_clean$InternetService == "Fiber optic", 1, 0)
telco_clean$InternetService_No <- ifelse(telco_clean$InternetService == "No", 1, 0)

logistic_model <- glm(ChurnBinary ~ tenure + Contract_OneYear + Contract_TwoYear + 
                        InternetService_Fiber + InternetService_No,
                      data = telco_clean,
                      family = binomial(link = "logit"))

# Model Summary:
summary_logistic <- summary(logistic_model)
print(summary_logistic)

# Odds Ratios:
coeffs_log <- coef(logistic_model)
odds_ratios <- exp(coeffs_log)
odds_ratios_ci <- exp(confint(logistic_model))
or_table <- data.frame(
  OddsRatio = odds_ratios,
  CI_Lower = odds_ratios_ci[, 1],
  CI_Upper = odds_ratios_ci[, 2]
)
print(round(or_table, 4))

# Coefficient Interpretations:
cat(sprintf("Tenure: OR = %.4f - Each month reduces churn odds by %.1f%%\n", 
            odds_ratios[2], (1 - odds_ratios[2]) * 100))
cat(sprintf("One-Year Contract: OR = %.4f - Reduces churn odds by %.1f%% vs month-to-month\n", 
            odds_ratios[3], (1 - odds_ratios[3]) * 100))
cat(sprintf("Two-Year Contract: OR = %.4f - Reduces churn odds by %.1f%% vs month-to-month\n", 
            odds_ratios[4], (1 - odds_ratios[4]) * 100))
cat(sprintf("Fiber Internet: OR = %.4f - Increases churn odds by %.1f%% vs DSL\n", 
            odds_ratios[5], (odds_ratios[5] - 1) * 100))

# Predictions and evaluation
predicted_probs <- predict(logistic_model, type = "response")
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)

# Confusion Matrix:
conf_matrix <- confusionMatrix(factor(predicted_class), 
                               factor(telco_clean$ChurnBinary),
                               positive = "1")
print(conf_matrix)

# Model Performance Metrics:
cat(sprintf("Accuracy: %.4f (%.2f%%)\n", 
            conf_matrix$overall['Accuracy'],
            conf_matrix$overall['Accuracy'] * 100))
cat(sprintf("Sensitivity: %.4f\n", conf_matrix$byClass['Sensitivity']))
cat(sprintf("Specificity: %.4f\n", conf_matrix$byClass['Specificity']))
cat(sprintf("Precision: %.4f\n", conf_matrix$byClass['Pos Pred Value']))

# ROC Curve
roc_obj <- roc(telco_clean$ChurnBinary, predicted_probs)
cat(sprintf("AUC: %.4f\n", auc(roc_obj)))

# Visualization
png("plots/11_logistic_regression_results.png", width = 1200, height = 800)
par(mfrow = c(2, 2))

# ROC Curve
plot(roc_obj, main = "ROC Curve - Logistic Regression",
     col = "blue", lwd = 2)
text(0.6, 0.3, sprintf("AUC = %.3f", auc(roc_obj)), cex = 1.5)

# Confusion Matrix
conf_mat_table <- table(Predicted = predicted_class, Actual = telco_clean$ChurnBinary)
barplot(conf_mat_table, beside = TRUE, 
        main = "Confusion Matrix", xlab = "Actual Churn Status", ylab = "Count",
        col = c("lightgreen", "lightcoral"),
        legend = c("Predicted: No Churn", "Predicted: Churn"),
        args.legend = list(x = "topright"))

# Predicted Probabilities
hist(predicted_probs, breaks = 50,
     main = "Distribution of Predicted Churn Probabilities",
     xlab = "Predicted Probability of Churn",
     col = "skyblue", border = "white")
abline(v = 0.5, col = "red", lwd = 2, lty = 2)

# Odds Ratios
or_plot <- odds_ratios[-1]
names(or_plot) <- c("Tenure", "Contract: 1-Year", "Contract: 2-Year", 
                    "Internet: Fiber", "Internet: None")
barplot(or_plot, main = "Odds Ratios (Churn Predictors)",
        ylab = "Odds Ratio", col = ifelse(or_plot < 1, "green", "red"),
        las = 2, cex.names = 0.8)
abline(h = 1, col = "black", lwd = 2, lty = 2)
dev.off()

#  Phase 3 Complete
#  All three regression models developed

################################################################################
# PHASE 4: MODEL EVALUATION & OPTIMIZATION
################################################################################

# 4.1 TRAIN-TEST SPLIT
train_index <- createDataPartition(telco_clean$CLV, p = 0.70, list = FALSE)
train_data <- telco_clean[train_index, ]
test_data <- telco_clean[-train_index, ]

cat(sprintf("Training Set: %d observations (%.1f%%)\n", 
            nrow(train_data), 100 * nrow(train_data) / nrow(telco_clean)))
cat(sprintf("Test Set: %d observations (%.1f%%)\n", 
            nrow(test_data), 100 * nrow(test_data) / nrow(telco_clean)))


# 4.2 EVALUATE SIMPLE LINEAR REGRESSION
simple_model_train <- lm(CLV ~ tenure, data = train_data)
train_pred_simple <- predict(simple_model_train, train_data)
test_pred_simple <- predict(simple_model_train, test_data)

calc_metrics <- function(actual, predicted, model_name = "Model") {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  return(data.frame(
    Model = model_name, MSE = mse, RMSE = rmse, MAE = mae,
    MAPE = mape, R_squared = r_squared
  ))
}
metrics_simple_train <- calc_metrics(train_data$CLV, train_pred_simple, 
                                     "Simple LR (Train)")
metrics_simple_test <- calc_metrics(test_data$CLV, test_pred_simple, 
                                    "Simple LR (Test)")

print(metrics_simple_train)  # Training Set Metrics:
print(metrics_simple_test)   # Test Set Metrics:

r2_diff <- abs(metrics_simple_train$R_squared - metrics_simple_test$R_squared)
cat(sprintf("\nR² Difference: %.4f ", r2_diff))
cat(ifelse(r2_diff < 0.05, "✓ No overfitting\n", "⚠ Possible overfitting\n"))

# Assumption testing
shapiro_test <- shapiro.test(sample(simple_model_train$residuals, 5000))
bp_test <- bptest(simple_model_train)
cat(sprintf("\nNormality test p-value: %.4f\n", shapiro_test$p.value))
cat(sprintf("Homoscedasticity test p-value: %.4f\n", bp_test$p.value))


# 4.3 EVALUATE & OPTIMIZE MULTIPLE REGRESSION

multiple_model_train <- lm(CLV ~ tenure + MonthlyCharges + ServiceAdoptionScore, 
                           data = train_data)
train_pred_multiple <- predict(multiple_model_train, train_data)
test_pred_multiple <- predict(multiple_model_train, test_data)

metrics_multiple_train <- calc_metrics(train_data$CLV, train_pred_multiple, 
                                       "Multiple LR (Train)")
metrics_multiple_test <- calc_metrics(test_data$CLV, test_pred_multiple, 
                                      "Multiple LR (Test)")

# Original Multiple Regression:
print(metrics_multiple_train)   # Training Set:
print(metrics_multiple_test)    # Test Set:

# Stepwise optimization
full_model <- lm(CLV ~ tenure + MonthlyCharges + ServiceAdoptionScore + 
                   StabilityIndex + RevenueCapacity, data = train_data)
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)

# Optimized Model (Stepwise Selection 
print(names(coef(step_model)))   #Variables Selected:

test_pred_optimized <- predict(step_model, test_data)
metrics_optimized_test <- calc_metrics(test_data$CLV, test_pred_optimized, 
                                       "Optimized Multiple LR (Test)")

print(metrics_optimized_test)       # Optimized Model Test Performance:
print(vif(step_model))              # VIF Check:


# 4.4 EVALUATE & OPTIMIZE LOGISTIC REGRESSION
cat("\n--- 4.4 Logistic Regression Evaluation & Optimization ---\n")

# Prepare data
train_data$Contract_TwoYear <- ifelse(train_data$Contract == "Two year", 1, 0)
train_data$Contract_OneYear <- ifelse(train_data$Contract == "One year", 1, 0)
train_data$InternetService_Fiber <- ifelse(train_data$InternetService == "Fiber optic", 1, 0)
train_data$InternetService_No <- ifelse(train_data$InternetService == "No", 1, 0)

test_data$Contract_TwoYear <- ifelse(test_data$Contract == "Two year", 1, 0)
test_data$Contract_OneYear <- ifelse(test_data$Contract == "One year", 1, 0)
test_data$InternetService_Fiber <- ifelse(test_data$InternetService == "Fiber optic", 1, 0)
test_data$InternetService_No <- ifelse(test_data$InternetService == "No", 1, 0)

logistic_model_train <- glm(ChurnBinary ~ tenure + Contract_OneYear + Contract_TwoYear + 
                              InternetService_Fiber + InternetService_No,
                            data = train_data, family = binomial(link = "logit"))

evaluate_logistic <- function(model, data, threshold = 0.5, set_name = "Data") {
  pred_probs <- predict(model, data, type = "response")
  pred_class <- ifelse(pred_probs > threshold, 1, 0)
  
  cm <- confusionMatrix(factor(pred_class, levels = c(0, 1)), 
                        factor(data$ChurnBinary, levels = c(0, 1)),
                        positive = "1")
  roc_obj <- roc(data$ChurnBinary, pred_probs, quiet = TRUE)
  
  cat(sprintf("\n=== %s Performance ===\n", set_name))
  cat(sprintf("Accuracy: %.4f (%.2f%%)\n", cm$overall['Accuracy'], 
              cm$overall['Accuracy'] * 100))
  cat(sprintf("Sensitivity: %.4f\n", cm$byClass['Sensitivity']))
  cat(sprintf("Specificity: %.4f\n", cm$byClass['Specificity']))
  cat(sprintf("F1-Score: %.4f\n", cm$byClass['F1']))
  cat(sprintf("AUC: %.4f\n", auc(roc_obj)))
  
  return(list(metrics = data.frame(
    Accuracy = cm$overall['Accuracy'], Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'], F1 = cm$byClass['F1'],
    AUC = auc(roc_obj)), roc = roc_obj))
}

test_results <- evaluate_logistic(logistic_model_train, test_data, 
                                  set_name = "Test Set (Original)")

# Threshold optimization
coords_obj <- coords(test_results$roc, "best", ret = "threshold", 
                     best.method = "youden")
optimal_threshold <- coords_obj[[1]]
cat(sprintf("\nOptimal Threshold: %.4f\n", optimal_threshold))

test_results_optimized <- evaluate_logistic(logistic_model_train, test_data, 
                                            threshold = optimal_threshold,
                                            set_name = "Test Set (Optimized Threshold)")

# Feature optimization
full_logistic <- glm(ChurnBinary ~ tenure + MonthlyCharges + 
                       Contract_OneYear + Contract_TwoYear + 
                       InternetService_Fiber + InternetService_No +
                       ServiceAdoptionScore, data = train_data,
                     family = binomial(link = "logit"))

step_logistic <- stepAIC(full_logistic, direction = "both", trace = FALSE)
print(names(coef(step_logistic)))         # Optimized Model Variables:

test_results_final <- evaluate_logistic(step_logistic, test_data, 
                                        set_name = "Test Set (Optimized Features)")

# 4.5 COMPREHENSIVE MODEL COMPARISON

clv_comparison <- rbind(metrics_simple_test, metrics_multiple_test, 
                        metrics_optimized_test)
print(clv_comparison)       # CLV Prediction Models (Test Set):

best_clv_idx <- which.min(clv_comparison$RMSE)
cat(sprintf("\nBest CLV Model: %s\n", clv_comparison$Model[best_clv_idx]))
cat(sprintf("  Lowest RMSE: $%.2f\n", clv_comparison$RMSE[best_clv_idx]))
cat(sprintf("  Highest R²: %.4f\n", clv_comparison$R_squared[best_clv_idx]))

# ------------------------------------------------------------------------------
# 4.6 VISUALIZATIONS
# ------------------------------------------------------------------------------

cat("\n--- 4.6 Creating Evaluation Visualizations ---\n")

# Model Comparison
png("plots/12_model_comparison.png", width = 1200, height = 800)
par(mfrow = c(2, 2))

barplot(clv_comparison$RMSE, names.arg = c("Simple", "Multiple", "Optimized"),
        main = "CLV Models: RMSE (Lower is Better)", ylab = "RMSE ($)",
        col = c("lightblue", "steelblue", "darkblue"),
        ylim = c(0, max(clv_comparison$RMSE) * 1.2))
text(x = c(0.7, 1.9, 3.1), y = clv_comparison$RMSE + 50,
     labels = sprintf("$%.0f", clv_comparison$RMSE))

barplot(clv_comparison$R_squared, names.arg = c("Simple", "Multiple", "Optimized"),
        main = "CLV Models: R² (Higher is Better)", ylab = "R-squared",
        col = c("lightgreen", "green", "darkgreen"), ylim = c(0, 1))
text(x = c(0.7, 1.9, 3.1), y = clv_comparison$R_squared + 0.05,
     labels = sprintf("%.3f", clv_comparison$R_squared))

dev.off()

# Actual vs Predicted
png("plots/13_actual_vs_predicted.png", width = 1000, height = 600)
par(mfrow = c(1, 2))

plot(test_data$CLV, test_pred_optimized,
     main = "Actual vs Predicted CLV", xlab = "Actual CLV ($)",
     ylab = "Predicted CLV ($)", col = alpha("blue", 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2, lty = 2)
legend("topleft", 
       legend = c(sprintf("R² = %.3f", metrics_optimized_test$R_squared),
                  sprintf("RMSE = $%.0f", metrics_optimized_test$RMSE)),
       bty = "n")

residuals_opt <- test_data$CLV - test_pred_optimized
plot(test_pred_optimized, residuals_opt,
     main = "Residual Plot", xlab = "Predicted CLV ($)",
     ylab = "Residuals ($)", col = alpha("darkgreen", 0.5), pch = 16)
abline(h = 0, col = "red", lwd = 2, lty = 2)
dev.off()

# ROC Comparison
png("plots/14_roc_comparison.png", width = 800, height = 800)
plot(test_results$roc, col = "blue", lwd = 2,
     main = "ROC Curves: Churn Prediction Models")
lines(test_results_optimized$roc, col = "green", lwd = 2)
lines(test_results_final$roc, col = "red", lwd = 2)
legend("bottomright",
       legend = c(sprintf("Original (AUC=%.3f)", test_results$metrics$AUC),
                  sprintf("Opt. Threshold (AUC=%.3f)", test_results_optimized$metrics$AUC),
                  sprintf("Opt. Features (AUC=%.3f)", test_results_final$metrics$AUC)),
       col = c("blue", "green", "red"), lwd = 2)
dev.off()

# ------------------------------------------------------------------------------
# 4.7 SAVE RESULTS
# ------------------------------------------------------------------------------

write.csv(clv_comparison, "results/clv_models_comparison.csv", row.names = FALSE)
saveRDS(step_model, "results/final_clv_model.rds")
saveRDS(step_logistic, "results/final_churn_model.rds")

# Phase 4 Complete
# All results saved in results/ directory

################################################################################
# FINAL SUMMARY
################################################################################

# Summary of generated files and assets: 
# Data: telco_customer_churn_cleaned.csv 
# Plots (14 total): # 01-08: EDA visualizations (Phase 2) 
# 09-11: Model development plots (Phase 3) 
# 12-14: Evaluation plots (Phase 4) 
# Results: clv_models_comparison.csv, final_clv_model.rds, final_churn_model.rds


cat("Best Models:\n")
cat(sprintf("  CLV Prediction: Optimized Multiple Regression (R² = %.4f, RMSE = $%.2f)\n",
            metrics_optimized_test$R_squared, metrics_optimized_test$RMSE))
cat(sprintf("  Churn Prediction: Optimized Logistic (Accuracy = %.2f%%, AUC = %.4f)\n",
            test_results_final$metrics$Accuracy * 100, test_results_final$metrics$AUC))

# All phases complete. Ready for final report compilation.

