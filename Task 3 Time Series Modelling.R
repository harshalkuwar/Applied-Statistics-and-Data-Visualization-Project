# Load Required Libraries
library(readxl)      # For reading Excel files
library(forecast)    # For time series modeling
library(tseries)     # For stationarity tests
library(ggplot2)     # For visualization

# Step 1: Load the data
data <- read_excel("Marriage.xlsx")
colnames(data)

str(data$`UK`)  # Check the structure of the column


# Step 2: Inspect Data
str(data)
head(data)

# Step 3: Check Missing Values
missing_values <- colSums(is.na(data))
print("Missing Values:")
print(data)
print(missing_values)

# Step 3: Clean  Missing Values

data[data == ":"] <- NA
data[data == "N/A"] <- NA

# Convert columns to numeric, filling missing values with mean
data <- data.frame(lapply(data, function(x) {
  if (is.numeric(as.numeric(as.character(x)))) {
    x <- as.numeric(as.character(x))  # Convert to numeric
    ifelse(is.na(x), mean(x, na.rm = TRUE), x)  # Replace NAs with mean
  } else {
    x  # Leave non-numeric columns as-is
  }
}))

print("After Cleaning Missing Values:")
print(data)
sum(is.na(data$`UK`))



# Step 4: Initial Exploration - Visualizing the Time Series
years <- data$Year
marriage_ts <- ts(data$`UK`, start = min(years), frequency = 1) 
plot(marriage_ts, main = "UK Marriages Over Time", ylab = "Marriages", xlab = "Year")


#  Decomposition

# a. Non-seasonal decomposition

# Set the frequency: 12 for monthly, 4 for quarterly, or 1 for annual
marriage_ts <- ts(data$`UK`, start = min(data$Year), frequency = 1)

# Handle decomposition based on the data's nature
if (frequency(marriage_ts) > 1) {
  # Seasonal decomposition using STL for seasonal data
  stl_decomp <- stl(marriage_ts, s.window = "periodic")
  plot(stl_decomp)
} else {
  # Non-seasonal decomposition for annual data
  trend <- stats::filter(marriage_ts, filter = rep(1/3, 3), sides = 2)
  irregular <- marriage_ts - trend
  
  # Plot results
  par(mfrow = c(3, 1))
  plot(marriage_ts, main = "Original Time Series")
  plot(trend, main = "Trend Component")
  plot(irregular, main = "Irregular Component")
}






# b. Seasonal decomposition
# Create the time series object
marriage_ts <- ts(data$`UK`, start = min(data$Year), frequency = 1)  

# Decomposition logic
if (frequency(marriage_ts) > 1) {
  # Use decompose or stl for seasonal data
  seasonal_decomp <- decompose(marriage_ts, type = "multiplicative") 
} else {
  # Non-seasonal decomposition for annual data
  trend <- stats::filter(marriage_ts, filter = rep(1/3, 3), sides = 2)
  irregular <- marriage_ts - trend
  
  # Plot the results
par(mfrow = c(1, 1))
plot(trend, main = "Trend Component")
}




# c. Seasonal Adjustment
# Calculate the trend using a moving average
trend <- stats::filter(marriage_ts, filter = rep(1/3, 3), sides = 2)

# Seasonally adjusted series (remove the trend component)
seasonally_adjusted <- marriage_ts - trend

# Plot the original series and the adjusted series
par(mfrow = c(2, 1))
plot(marriage_ts, main = "Original Time Series", ylab = "Marriages")
plot(seasonally_adjusted, main = "Seasonally Adjusted Series", ylab = "Adjusted Marriages")

# Time series modelling
# 1)  Exponential Smoothing Techniques
ets_model <- ets(marriage_ts)
summary(ets_model)

# Forecast with Exponential Smoothing
ets_forecast <- forecast(ets_model, h = 5)  # Forecast for 5 years
plot(ets_forecast)

#  Generate a histogram
hist(data$UK, breaks = 20, main = "Histogram of Number of Marriages",
     xlab = "Number of Marriages", col = "lightblue", border = "black")

# 2)  ARIMA Models
# Check stationarity
adf_test <- adf.test(marriage_ts, alternative = "stationary")
print(adf_test)

# If not stationary, differencing
diff_marriage_ts <- diff(marriage_ts)
plot(diff_marriage_ts, main = "Differenced Series", ylab = "Differenced Marriages")

# Fit ARIMA Model
auto_arima_model <- auto.arima(marriage_ts)
summary(auto_arima_model)
# Forecast with ARIMA
arima_forecast <- forecast(auto_arima_model, h = 5)
plot(arima_forecast)




 # 3 Holt winter method
# Step 6: Holt-Winters Method
# Apply Holt-Winters method
hw_model <- HoltWinters(marriage_ts, seasonal = "additive")  # Change to "multiplicative" if needed
summary(hw_model)

# Forecast with Holt-Winters
hw_forecast <- forecast(hw_model, h = 5)  # Forecast for 5 years
plot(hw_forecast, main = "Holt-Winters Forecast for UK Marriages")


# residuals plot
residuals <- hw_forecast$residuals
plot(residuals, main = "Residuals of Holt-Winters Model", xlab = "Time", ylab = "Residuals", type = "o")
abline(h = 0, col = "red", lty = 2)










# 3. Modelling with Forecasting and Accuracy Measurement

# Split the Data into Training and Test Sets
train_size <- floor(0.8 * length(marriage_ts))
train_ts <- window(marriage_ts, end = c(time(marriage_ts)[train_size]))
test_ts <- window(marriage_ts, start = c(time(marriage_ts)[train_size + 1]))

# 1. Fit the ETS Model on the Training Set
ets_model <- ets(train_ts)

# Forecast Using the ETS Model
ets_forecast <- forecast(ets_model, h = length(test_ts))

# Calculate Accuracy Metrics for the ETS Model
ets_accuracy <- accuracy(ets_forecast, test_ts)
mae_ets <- ets_accuracy["Test set", "MAE"]
rmse_ets <- ets_accuracy["Test set", "RMSE"]
sse_ets <- sum((ets_forecast$mean - test_ts)^2, na.rm = TRUE)

# 2. Fit the ARIMA Model on the Training Set
arima_model <- auto.arima(train_ts)

# Forecast Using the ARIMA Model
arima_forecast <- forecast(arima_model, h = length(test_ts))

# Calculate Accuracy Metrics for the ARIMA Model
arima_accuracy <- accuracy(arima_forecast, test_ts)
mae_arima <- arima_accuracy["Test set", "MAE"]
rmse_arima <- arima_accuracy["Test set", "RMSE"]
sse_arima <- sum((arima_forecast$mean - test_ts)^2, na.rm = TRUE)

# 3. Fit the Holt-Winters Model on the Training Set
hw_model <- HoltWinters(train_ts, seasonal = "additive")  # Change to "multiplicative" if needed

# Forecast Using the Holt-Winters Model
hw_forecast <- forecast(hw_model, h = length(test_ts))

# Calculate Accuracy Metrics for the Holt-Winters Model
hw_accuracy <- accuracy(hw_forecast, test_ts)
mae_hw <- hw_accuracy["Test set", "MAE"]
rmse_hw <- hw_accuracy["Test set", "RMSE"]
sse_hw <- sum((hw_forecast$mean - test_ts)^2, na.rm = TRUE)

# Print the Comparison of Models
cat("Comparison of Models:\n")
cat(sprintf("ETS:    MAE = %.3f, RMSE = %.3f, SSE = %.3f\n", mae_ets, rmse_ets, sse_ets))
cat(sprintf("ARIMA:  MAE = %.3f, RMSE = %.3f, SSE = %.3f\n", mae_arima, rmse_arima, sse_arima))
cat(sprintf("Holt-Winters:  MAE = %.3f, RMSE = %.3f, SSE = %.3f\n", mae_hw, rmse_hw, sse_hw))

# Visualization: Add Holt-Winters Metrics
metrics <- data.frame(
  Metric = c("MAE", "RMSE", "SSE"),
  ETS = c(mae_ets, rmse_ets, sse_ets),
  ARIMA = c(mae_arima, rmse_arima, sse_arima),
  `Holt-Winters` = c(mae_hw, rmse_hw, sse_hw)
)


# Create a line graph
library(ggplot2)
ggplot(metrics_long, aes(x = Metric, y = Value, group = Model, color = Model)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Line Graph of ETS, ARIMA, and Holt-Winters Metrics",
       x = "Metric Type",
       y = "Value",
       color = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




