# 🚗 Driver Behavior Prediction System

A Machine Learning-based system that analyzes sensor and driving data to classify different driving behaviors and identify potentially unsafe driving events.

## 📌 Project Overview

The system uses Machine Learning to analyze driving patterns and predict the type of driving event occurring. It classifies driver behavior into four categories:

* Sudden Acceleration
* Sudden Right Turn
* Sudden Left Turn
* Sudden Brake

The project compares multiple Machine Learning algorithms to determine which model performs best for classifying these driving behaviors.

## 🎯 Objectives

* Analyze driving and sensor data.
* Extract meaningful features related to driver behavior.
* Classify different driving events using Machine Learning.
* Compare the performance of multiple ML models.
* Identify the best-performing model based on classification accuracy.
* Analyze the model's predictions using classification reports and confusion matrices.

## 🔄 Machine Learning Workflow

The project follows these main steps:

1. Load the driving dataset.
2. Check for missing values and duplicate records.
3. Remove duplicate records.
4. Create meaningful behavior labels.
5. Perform feature engineering using acceleration measurements.
6. Split the data into training and testing sets.
7. Scale the features using `StandardScaler`.
8. Train multiple Machine Learning models.
9. Compare model accuracy.
10. Evaluate the best-performing model using a classification report and confusion matrix.
11. Analyze important features when supported by the selected model.

## 🤖 Machine Learning Models

Five classification algorithms are compared:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Classifier (SVC) using RBF kernel
* Random Forest
* XGBoost

The model with the highest test accuracy is selected as the best-performing model.

## 📊 Data & Features

The dataset contains sensor measurements related to vehicle and driver behavior.

Feature engineering is also applied by calculating the total acceleration magnitude from the X, Y, and Z acceleration measurements.

The target variable represents the type of driving event:

| Target | Behavior            |
| ------ | ------------------- |
| 1      | Sudden Acceleration |
| 2      | Sudden Right Turn   |
| 3      | Sudden Left Turn    |
| 4      | Sudden Brake        |

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

A normalized confusion matrix is also generated to better understand the classification performance for each driving behavior.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Google Colab

## 👩‍💻 My Role

**Machine Learning Developer**

Responsible for the Machine Learning component, including data preprocessing, feature engineering, applying and comparing classification models, and analyzing their prediction results.

## 🚀 Future Improvements

* Integrate the prediction model into an interactive dashboard.
* Add real-time driving behavior prediction.
* Expand the number of detectable driving events.
* Improve the system using additional sensor and driving features.
