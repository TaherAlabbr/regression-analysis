# ==============================================================================
# Heart Disease Analysis - Logistic Regression Framework
# ==============================================================================

# 1. SETUP & LIBRARY LOADING
# ------------------------------------------------------------------------------
# Install missing packages if necessary
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, car, broom, pROC, caret)

# 2. DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------------
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

col_names <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num")

heart_data <- read.csv(url, header = FALSE, col.names = col_names, na.strings = "?")

# Preprocessing Pipeline
clean_data <- heart_data %>%
  drop_na() %>%
  mutate(
    # Create Binary Target: 0 = No Disease, 1 = Disease
    target = ifelse(num > 0, 1, 0),
    
    # Factor encoding (Explicit labels help with plot readability)
    sex     = factor(sex, labels = c("Female", "Male")),
    cp      = as.factor(cp),
    fbs     = as.factor(fbs),
    restecg = as.factor(restecg),
    exang   = as.factor(exang),
    slope   = as.factor(slope),
    thal    = as.factor(thal),
    ca      = as.numeric(ca)
  )

# 3. MODEL SPECIFICATION
# ------------------------------------------------------------------------------

# --- Model 1: Reduced (Consumer/Wearable) ---
# Variables: Age, Sex, HR, BP, Chest Pain, Exercise Angina
model_reduced <- glm(target ~ age + sex + thalach + trestbps + cp + exang, 
                     data = clean_data, 
                     family = binomial(link = "logit"))

# --- Model 2: Full (Clinical/Invasive) ---
# Adds: ECG, ST depression, slope, cholesterol, fasting sugar, fluoroscopy, thallium
model_full <- glm(target ~ age + sex + thalach + trestbps + cp + exang + 
                    restecg + oldpeak + slope + chol + fbs + ca + thal, 
                  data = clean_data, 
                  family = binomial(link = "logit"))

# Calculate McFadden's Pseudo R-Squared
calc_r2 <- function(model) {
  1 - (model$deviance / model$null.deviance)
}

cat("\n--- Model Fit (McFadden's Pseudo R2) ---\n")
print(paste("Consumer Model R2:", round(calc_r2(model_reduced), 3)))
print(paste("Clinical Model R2:", round(calc_r2(model_full), 3)))

# 4. STATISTICAL INFERENCE
# ------------------------------------------------------------------------------
cat("\n--- Likelihood Ratio Test (Nested Model Comparison) ---\n")
# Testing if the clinical variables add significant information
print(anova(model_reduced, model_full, test = "Chisq"))

cat("\n--- AIC Comparison ---\n")
# Lower AIC indicates better model fit penalizing for complexity
model_stats <- bind_rows(
  glance(model_reduced) %>% mutate(Model = "Reduced (Consumer)"),
  glance(model_full) %>% mutate(Model = "Full (Clinical)")
) %>% select(Model, AIC, BIC, deviance)

print(model_stats)

# 5. DIAGNOSTICS
# ------------------------------------------------------------------------------

# A. Multicollinearity (VIF)
cat("\n--- Variance Inflation Factors (VIF) ---\n")
# Check if predictors are correlated (VIF > 5 is concerning)
print(vif(model_full))

# B. Influential Observations (Cook's Distance)
par(mfrow = c(1, 1)) # Reset plot window
plot(model_reduced, which = 4, main = "Cook's Distance (Outlier Check)")
abline(h = 4/nrow(clean_data), col = "red", lty = 2) 

# C. Linearity of the Logit (Visual Inspection)
# ------------------------------------------------------------------------------
# We check if continuous predictors are linearly related to the log-odds
probabilities <- predict(model_full, type = "response")
probabilities <- pmin(pmax(probabilities, 0.0001), 0.9999) 
logits <- log(probabilities / (1 - probabilities))

# Modern reshaping with pivot_longer
linearity_data <- clean_data %>%
  dplyr::select(age, thalach, trestbps, oldpeak) %>%
  mutate(logit = logits) %>%
  pivot_longer(cols = -logit, names_to = "predictors", values_to = "predictor_value")

# Plotting
linearity_plot <- ggplot(linearity_data, aes(x = predictor_value, y = logit)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", color = "blue", se = FALSE) + 
  facet_wrap(~ predictors, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Linearity Assumption Check",
    subtitle = "Blue line should be roughly linear",
    y = "Log-Odds (Logit)",
    x = "Predictor Value"
  )

print(linearity_plot)

# 6. PERFORMANCE METRICS (ROC & CONFUSION MATRIX)
# ------------------------------------------------------------------------------

# A. ROC Curve Analysis
prob_reduced <- predict(model_reduced, type = "response")
prob_full <- predict(model_full, type = "response")

roc_reduced <- roc(clean_data$target, prob_reduced, quiet = TRUE)
roc_full <- roc(clean_data$target, prob_full, quiet = TRUE)

plot(roc_full, col = "blue", main = "ROC Comparison: Clinical vs. Consumer")
plot(roc_reduced, col = "red", add = TRUE)
legend("bottomright", legend = c(paste("Clinical AUC:", round(auc(roc_full), 3)), 
                                 paste("Consumer AUC:", round(auc(roc_reduced), 3))),
       col = c("blue", "red"), lty = 1)

# B. Confusion Matrix (Threshold = 0.5)
predicted_class <- ifelse(prob_reduced > 0.5, 1, 0)
conf_matrix <- confusionMatrix(factor(predicted_class), factor(clean_data$target), positive = "1")

cat("\n--- Consumer Model Performance Metrics ---\n")
print(conf_matrix$byClass[c("Sensitivity", "Specificity", "Precision", "Recall")])
print(conf_matrix$overall["Accuracy"])

# 7. MODEL INTERPRETATION
# ------------------------------------------------------------------------------
cat("\n--- Key Predictors (Odds Ratios) ---\n")
# Using broom::tidy to get a clean table of coefficients
tidy(model_reduced, exponentiate = TRUE, conf.int = TRUE) %>% 
  filter(p.value < 0.05) %>% 
  select(term, estimate, p.value, conf.low, conf.high) %>% 
  print()