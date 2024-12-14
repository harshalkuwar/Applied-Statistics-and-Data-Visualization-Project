


# 1. Load libraries and Dataset 

# A) Load libraries
# Load necessary libraries

install.packages("glmnet")
install.packages("randomForest")
install.packages("Metrics")
library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(car)
library(lmtest)
library(glmnet)
library(randomForest)
library(Metrics)
library(reshape2)

# B) Read the dataset
data <- read_excel("concrete compressive strength.xlsx")



# 2. Exploratory data analysis

# View the structure of the dataset
str(data)

# Compute summary statistics
summary(data)


# Rename columns to simpler names

colnames(data) <- make.names(colnames(data))

# Check updated column names
colnames(data)

# Histograms for numeric columns
numeric_cols <- names(data)[sapply(data, is.numeric)]
for (col in numeric_cols) {
  print(ggplot(data, aes(x = .data[[col]])) +
          geom_histogram(binwidth = 30, fill = "blue", color = "black", alpha = 0.7) +
          ggtitle(paste("Histogram of", col)) +
          theme_minimal())
}


# Boxplots to Explore Distributions by Concrete Category
numeric_cols <- names(data)[sapply(data, is.numeric)]  # Identify numeric columns

for (col in numeric_cols) {
  print(
    ggplot(data, aes(x = .data[[categorical_col]], y = .data[[col]], fill = .data[[categorical_col]])) +
      geom_boxplot(alpha = 0.7) +
      ggtitle(paste("Boxplot of", col, "by", categorical_col)) +
      scale_fill_manual(values = c("Coarse" = "pink", "Fine" = "orange")) +  # Assign colors
      theme_minimal() +
      labs(fill = "Concrete Category")  # Add legend label
  )
}



# Scatterplot examples
ggplot(data, aes(x = Cement..component.1..kg.in.a.m.3.mixture., 
                 y = Concrete.compressive.strength.MPa..megapascals.)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  ggtitle("Cement vs Concrete Compressive Strength") +
  theme_minimal()



# Correlation heatmap
cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper")



# 3) Data Preprocessing

#1. Check for missing values
colSums(is.na(data))


# Ensure column names are correct and unique
colnames(data) <- make.unique(colnames(data))

# Apply log transformation for numeric columns containing "component"
data <- data %>%
  mutate(across(contains("component"), ~ ifelse(. > 0, log(.), .)))


# Encoding categorical variables

data <- data %>%
  mutate(
    Concrete.Category = as.numeric(factor(Concrete.Category)),
    Contains.Fly.Ash = as.numeric(Contains.Fly.Ash)
  )

# Define the target column

target_column <- "Concrete.compressive.strength.MPa..megapascals."

# Summarize the target variable to confirm its properties
summary(data[[target_column]])

# Check if the column exists and plot the histogram
if (target_column %in% colnames(data)) {
  # Plot the histogram
  ggplot(data, aes(x = .data[[target_column]])) +
    geom_histogram(binwidth = 5, fill = "Pink", color = "black", alpha = 0.7) +
    ggtitle("Histogram of Concrete Compressive Strength") +
    xlab("Compressive Strength (MPa)") +
    ylab("Frequency") +
    theme_minimal()
} else {
  print("Target variable not found in the dataset!")
}

# Q-Q Plot

# Standardize the data 
standardized_data <- scale(data$Concrete.compressive.strength.MPa..megapascals.)

# Generate Q-Q plot for standardized data
qqnorm(standardized_data, main = "Q-Q Plot (Standardized Data)")



# Breusch-Pagan Test

model <- lm(Concrete.compressive.strength.MPa..megapascals. ~ ., data = data)
bptest(model)


# One-sample K-S test
ks.test(
  data$Concrete.compressive.strength.MPa..megapascals., # Data sample
  "pnorm",                                              # Reference distribution
  mean = mean(data$Concrete.compressive.strength.MPa..megapascals., na.rm = TRUE),
  sd = sd(data$Concrete.compressive.strength.MPa..megapascals., na.rm = TRUE)
)


# 4. Correlation analysis

# Compute Pearson correlation
pearson_corr <- cor(numeric_columns, method = "pearson")

# Calculate the Pearson correlation matrix
pearson_corr <- cor(data %>% select_if(is.numeric), method = "pearson")


# Display the correlation matrix
print(pearson_corr)

# Melt the correlation matrix for ggplot2 visualization
melted_corr <- melt(pearson_corr)

# Visualize Pearson correlation as a heatmap with numbers
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +  # Create heatmap tiles
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), 
                       name = "Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 4) +  # Add numbers
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Pearson Correlation Heatmap with Values",
       x = "Variables",
       y = "Variables")



#  5. Regression 
#  Fit a multiple linear regression model
mlr_model <- lm(Concrete.compressive.strength.MPa..megapascals. ~ ., data = data)

# Summary of the model
summary(mlr_model)

# Check residual normality
qqnorm(residuals(mlr_model))
qqline(residuals(mlr_model), col = "red")

# Create a Q-Q plot using ggplot2
ggplot(data.frame(residuals = residuals(mlr_model)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  ggtitle("Q-Q Plot of Residuals") +
  theme_minimal()




# Advanced regression 

#1) Ridge regression

# Prepare data for Ridge Regression
x <- model.matrix(Concrete.compressive.strength.MPa..megapascals. ~ ., data = data)[, -1]
# Extract the dependent variable
y <- data$Concrete.compressive.strength.MPa..megapascals.

# Fit Ridge Regression
ridge_model <- glmnet(x, y, alpha = 0)  # alpha = 0 for Ridge
print(ridge_model)

# Plot the Ridge Regression coefficient paths
plot(ridge_model, xvar = "lambda", label = TRUE, main = "Ridge Regression Coefficient Paths")

# 2)  LASSO Regression
lasso_model <- glmnet(x, y, alpha = 1)  # alpha = 1 for LASSO
print(lasso_model)
# Plot the LASSO Regression coefficient paths
plot(lasso_model, xvar = "lambda", label = TRUE, main = "LASSO Regression Coefficient Paths")



#3) Random forest regression 

# Fit Random Forest Regression
rf_model <- randomForest(Concrete.compressive.strength.MPa..megapascals. ~ ., data = data, ntree = 100)

# Print model summary
print(rf_model)

# Variable importance
importance(rf_model)
varImpPlot(rf_model)

# Visualization

# Plot Cement Content vs. Predicted Strength (line)
plot(cement_seq, predictions, type = "l", col = "red", lwd = 2,
     xlab = "Cement Content",
     ylab = "Predicted Concrete Compressive Strength",
     main = "Random Forest Regression Line with Data Points")

# Add actual data points (scatter plot)
points(data$Cement..component.1..kg.in.a.m.3.mixture., 
       data$Concrete.compressive.strength.MPa..megapascals., 
       col = "blue", pch = 16, cex = 0.8)  # Blue points for original data


##Hypothesis Testing 

#1) Test for normality

#a) Shapiro-Wilk Test
#Check whether a specific variable is normally distributed:
  
# Shapiro-Wilk Test for normality
shapiro.test(data$Concrete.compressive.strength.MPa..megapascals.)

#2) T-Test
# a) One-sample T-test
t.test(data$Concrete.compressive.strength.MPa..megapascals., mu = 30)  

#b) Two sample T Test
# Subset data into two groups (e.g., by a categorical variable)
group1 <- data$Concrete.compressive.strength.MPa..megapascals.[data$Contains.Fly.Ash == 0]
group2 <- data$Concrete.compressive.strength.MPa..megapascals.[data$Contains.Fly.Ash == 1]

# Two-sample T-test
t.test(group1, group2, var.equal = TRUE)

#3) Chi-Square Test

# Create a contingency table for two categorical variables
table_data <- table(data$Concrete.Category, data$Contains.Fly.Ash)

# Perform Chi-Square Test
chisq.test(table_data)

#4) Non-Parametric Tests

# Wilcoxon Rank-Sum Test
wilcox.test(group1, group2)



#6) Evaluate The Model

#1. Multicollinearity Check (For MLR)

# Fit the multiple linear regression (MLR) model
mlr_model <- lm(Concrete.compressive.strength.MPa..megapascals. ~ ., data = data)

# Calculate VIF for the model
vif(mlr_model)


# 2. Model Comparison

# Calculate RMSE 
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Define a custom MAE function
mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

# Prepare data
target_column <- "Concrete.compressive.strength.MPa..megapascals."  # Adjust as necessary
X <- model.matrix(as.formula(paste(target_column, "~ .")), data = data)[, -1]
y <- data[[target_column]]


ridge_rmse <- rmse(y, ridge_predictions)
lasso_rmse <- rmse(y, lasso_predictions)
poly_rmse <- rmse(y, poly_predictions)
rf_rmse <- rmse(y, rf_predictions)
mlr_rmse <- rmse(y, mlr_predictions)



# 1. Multiple Linear Regression (MLR)
mlr_model <- lm(as.formula(paste(target_column, "~ .")), data = data)
mlr_predictions <- predict(mlr_model, newdata = data)
mlr_r2 <- summary(mlr_model)$r.squared
mlr_rmse <- rmse(y, mlr_predictions)
mlr_mae <- mae(y, mlr_predictions)



# 2. Ridge Regression
ridge_model <- glmnet(X, y, alpha = 0)  # alpha = 0 for Ridge
ridge_predictions <- predict(ridge_model, s = 0.01, newx = X)
ridge_r2 <- cor(ridge_predictions, y)^2
ridge_rmse <- rmse(y, ridge_predictions)
ridge_mae <- mae(y, ridge_predictions)

# 3. LASSO Regression
lasso_model <- glmnet(X, y, alpha = 1)  # alpha = 1 for LASSO
lasso_predictions <- predict(lasso_model, s = 0.01, newx = X)
lasso_r2 <- cor(lasso_predictions, y)^2
lasso_rmse <- rmse(y, lasso_predictions)
lasso_mae <- mae(y, lasso_predictions)


# 4. Random Forest Regression


rf_model <- randomForest(as.formula(paste(target_column, "~ .")), data = data, ntree = 100)
rf_predictions <- predict(rf_model, newdata = data)
rf_r2 <- cor(rf_predictions, y)^2
rf_rmse <- rmse(y, rf_predictions)
rf_mae <- mae(y, rf_predictions)


# Create model comparison table
model_comparison <- data.frame(
  Model = c("MLR", "Ridge", "LASSO", "Random Forest"),
  R_Squared = c(mlr_r2, ridge_r2, lasso_r2, rf_r2),
  RMSE = c(mlr_rmse, ridge_rmse, lasso_rmse, rf_rmse),
  MAE = c(mlr_mae, ridge_mae, lasso_mae, rf_mae)
)

# Print model comparison table
print(model_comparison)
