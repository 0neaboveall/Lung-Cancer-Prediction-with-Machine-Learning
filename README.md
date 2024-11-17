# Lung Cancer Prediction with Machine Learning

This project applies machine learning techniques to predict lung cancer using patient health data. By utilizing Naive Bayes and Decision Tree models, it enables early detection, improving survival rates and reducing healthcare costs. The findings demonstrate how predictive analytics can enhance cancer diagnosis and care.

## Overview

- **Business Need:** Lung cancer is one of the leading causes of cancer deaths worldwide, with survival rates significantly higher when detected early. Traditional diagnostic methods are expensive and time-consuming. This project aims to develop a data-driven, cost-effective predictive model for early detection.
- **Techniques Used:**
  - Principal Component Analysis (PCA) for dimensionality reduction.
  - Naive Bayes classifier for robust predictions with categorical data.
  - Decision Tree for interpretable and efficient modeling.

## Dataset

- **Data Source:** Lung cancer survey dataset.
- **Features:** Includes patient attributes such as gender, smoking habits, allergies, coughing, swallowing difficulty, and anxiety.
- **Data Preprocessing:**
  - Conversion of categorical variables to factors.
  - Scaling and centering of numerical variables for normalization.

## Goals

1. Build a machine learning model to predict lung cancer risk.
2. Identify key health factors contributing to lung cancer.
3. Evaluate model performance using metrics like accuracy, AUC, and ROC curves.

## Methodology

1. **Data Preprocessing:**
   - Categorical variables converted to factors.
   - Data normalized for consistent modeling.
2. **Dimensionality Reduction:**
   - PCA used to reduce input variables while retaining key information.
3. **Modeling:**
   - **Naive Bayes:** Trained on categorical data for robust predictions.
   - **Decision Tree:** Offers transparent and interpretable decision rules.
4. **Evaluation Metrics:**
   - Confusion matrix, accuracy, AUC, and ROC curves.

## Results

- **Naive Bayes Model:**
  - **Accuracy:** 88.7%
  - **AUC:** 0.896
  - Performs well in distinguishing lung cancer cases.
- **Decision Tree Model:**
  - **Accuracy:** 84%
  - **AUC:** 0.84
  - Highlights key predictors like allergies, coughing, and anxiety.
- **Key Insights:**
  - Early predictors include difficulty swallowing, coughing severity, and anxiety levels.

## Repository Structure

1. Clone the repository:
   ```bash
   git clone https://github.com/0neaboveall/Lung-Cancer-Prediction-with-Machine-Learning.git
   cd Lung-Cancer-Prediction-with-Machine-Learning


