# Applied-Statistics-and-Data-Visualization-Project

Overview
This project explores applied statistics and data visualization techniques across three major tasks:

Task 1 Interactive Dashboard Design: Creating a user-friendly dashboard for exploring global economic trends.
Task 2 Statistical Analysis: Using predictive modeling techniques to optimize concrete compressive strength.
Task 3 Time Series Modeling: Forecasting marriage trends in the UK using advanced time series models.

The project highlights data preparation, exploratory analysis, and model evaluation using statistical and machine learning methods.

Table of Contents
Tasks Overview
Technologies Used
Setup Instructions
Key Features
Results and Insights
Limitations and Recommendations
Author Information

Tasks Overview

Task 1: Interactive Dashboard Design
Title: Designing an Interactive Dashboard for Exploring Global Economic Trends
Objective: To create a visually intuitive dashboard that simplifies complex economic data for users.
Key Features:
Visualizations: Bar charts, pie charts, area charts, and slicers for interactivity.
KPIs: GDP growth rate, unemployment rate, population, investment, and government debt.
Interactive Features: Filters, dropdowns, and hierarchies for exploring data by country or region.
Outcome: A comprehensive, interactive dashboard that highlights economic disparities and trends.

Task 2: Statistical Analysis
Title: Optimizing Concrete Compressive Strength: Predictive Modeling with Machine Learning
Objective: To identify factors influencing concrete compressive strength and build predictive models.
Techniques Used:
Exploratory Analysis: Histograms, scatterplots, and heatmaps to identify relationships between variables.
Regression Models: Multiple Linear Regression, Ridge, LASSO, and Random Forest.
Model Evaluation: Random Forest emerged as the best model, explaining 91.39% of the variance with low error metrics.
Statistical Tests: T-tests, Chi-Square tests, and non-parametric tests to validate hypotheses.

Task 3: Time Series Modeling
Title: Forecasting Marriage Trends in the UK
Objective: To predict future marriage trends using historical data.
Models Used:
Exponential Smoothing (ETS): Best-performing model with the lowest errors (RMSE: 103,575).
ARIMA: Reasonably accurate but slightly less effective than ETS.
Holt-Winters: Struggled to fit the data, with the highest errors.
Outcome: ETS is recommended for reliable forecasting of marriage trends.

Technologies Used
Data Visualization Tools: Microsoft Power BI, Matplotlib, Seaborn
Statistical Analysis: R, Python
Machine Learning Models: Random Forest, Ridge, LASSO, ARIMA, ETS
Data Manipulation: Pandas, NumPy
Forecasting Tools: R's forecast package, Python's statsmodels

Setup Instructions
Data Preparation:
Load datasets in .csv or .xlsx formats.
Ensure clean and preprocessed data using provided scripts.
For time series analysis, ensure the dataset is chronological and has no missing dates.

Environment Setup:
Install required Python/R libraries:
bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
For R, install forecast, ggplot2, and related packages.

Run Notebooks/Scripts:
Execute provided scripts or Jupyter notebooks for statistical analysis and modeling.
Use Power BI or Excel for dashboard construction.

Key Features
Interactive Visualizations:
Slicers and dropdowns for filtering global economic data.
Clear visual hierarchy for easy interpretation of trends.

Predictive Modeling:
Regression models to optimize construction material properties.
Feature selection and hypothesis testing to ensure model reliability.

Time Series Forecasting:
Advanced models like ARIMA and ETS for accurate predictions.
Comparison of multiple models to identify the best-fit approach.

Results and Insights
Task 1:
Emerging economies show higher but volatile GDP growth compared to developed nations.
Population trends highlight demographic challenges in Europe vs. rapid growth in Asia and Africa.

Task 2:
Cement content strongly correlates with compressive strength, while water negatively impacts it.
Random Forest regression is the most accurate model, outperforming MLR and Ridge.

Task 3:
UK marriage trends are best forecasted using ETS, capturing patterns with minimal errors.
Historical events significantly influence spikes in marriage rates.

Limitations and Recommendations
Task 1:
Limitation: Lack of detailed, localized economic data.
Recommendation: Include political and social factors for a more comprehensive dashboard.

Task 2:
Limitation: High multicollinearity in predictors like water and fly ash.
Recommendation: Use advanced feature selection techniques like PCA.

Task 3:
Limitation: Holt-Winters model struggled with non-seasonal data.
Recommendation: Focus on ETS or ARIMA for similar forecasting tasks.
