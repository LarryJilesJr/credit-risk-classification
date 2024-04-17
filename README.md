# credit-risk-classification

# Loan Status Prediction Model

## Overview of the Analysis
The purpose of this analysis was to develop a machine learning model to predict the status of loan applications as either healthy or high-risk based on financial information provided in the dataset. The analysis is designed to assist a financial institution in automating the loan approval process and identifying potentially risky loans early.

The dataset contained various financial features such as loan amount, interest rate, debt-to-income ratio, employment length, and credit score, among others. The target variable to predict was the loan status, which had two categories: healthy (0) and high-risk (1).

Here's a summary of the stages of the machine learning process:

1. Data Exploration: I explored the dataset to understand its structure, checked for missing values, and examined the distribution of the target variable. I also performed basic statistical analysis to gain insights into the features.

2. Data Preprocessing: Preprocessing steps included handling missing values, encoding categorical variables, and scaling numerical features to ensure uniformity and compatibility with the machine learning algorithms.

3. Model Selection: I utilized the Logistic Regression machine learning algorithm for its interpretability and effectiveness in binary classification tasks.

4. Model Training and Evaluation: I split the data into training and testing sets, trained the models on the training data, and evaluated their performance using metrics such as accuracy, precision, recall, and F1-score.

5. Model Tuning: I fine-tuned hyperparameters of the selected models to optimize their performance and generalization capabilities.

## Results
*Machine Learning Model 1:
    *Classification Report:
        -Precision for healthy loans (label 0): 1.00
        -Precision for high-risk loans (label 1): 0.85
        -Recall for healthy loans (label 0): 0.99
        -Recall for high-risk loans (label 1): 0.91
        -F1-score for healthy loans (label 0): 1.00
        -F1-score for high-risk loans (label 1): 0.88
        -Accuracy: 0.99
        -Macro average precision: 0.92
        -Macro average recall: 0.95
        -Macro average F1-score: 0.94
        -Weighted average precision: 0.99
        -Weighted average recall: 0.99
        -Weighted average F1-score: 0.99
        -Total support: 19384

Accuracy Score: The overall accuracy of the model is 0.99, indicating a high level of correct predictions.

Precision Score:
Precision for healthy loans (label 0) is 1.00, indicating very few false positives.
Precision for high-risk loans (label 1) is 0.85, showing a small proportion of healthy loans misclassified as high-risk.

Recall Score:
Recall for healthy loans (label 0) is 0.99, indicating that the model correctly identifies most healthy loans.
Recall for high-risk loans (label 1) is 0.91, suggesting that the model captures a large proportion of high-risk loans.

## Summary
Performance Evaluation:
Among the models evaluated, the Logistic Regression model performed the best based on various performance metrics such as accuracy, precision, recall, and F1-score. Logistic Regression achieved high accuracy scores for both healthy (label 0) and high-risk (label 1) loans, showing its effectiveness in correctly predicting loan statuses.

Prediction Importance:
The importance of prediction depends on the problem we are trying to solve. In the context of loan approval, it is crucial to accurately predict both healthy and high-risk loans. However, misclassification of high-risk loans (false negatives) can have more critical consequences, as it may lead to granting loans to applicants who are likely to default.

Recommendation:
Based on the results, I recommend using the Logistic Regression model for predicting loan statuses. It demonstrates high accuracy and performs well in identifying both healthy and high-risk loans. Continuous monitoring and periodic updates should be implemented to ensure the model's performance remains optimal over time and adapts to changes in the lending environment.

---
**Source Data: 

Chat GPT Provider: OpenAI Model Version: GPT-3.5 Training Data: Diverse internet text Training Duration: Training duration was about 1-2 hours @article{openai2023, author = {OpenAI}, title = {ChatGPT: A Language Model by OpenAI}, year = {2023}, url = {https://www.openai.com}, }

Class Videos

Stackoverflow