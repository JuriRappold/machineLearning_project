# Installing Packages

`pip install -r requirements.txt`

# Steps
## 1. Cleaning Data
1. drop duplicates
2. drop missing, corrupted(impossible) rows
3. outliers (z-score > threshold)
   1. visualize outliers
   2. group them if they are many
   3. extract feature: high_energy_user (True/False) || class: HIGH_ENERGY_USER, LOW_ENERGY_USER
   4. play around w/ threshold

## 2. Data Preparation
1. Encoding the classes
2. Feature Extraction
   1. Energy Company (from the files)
   2. outlier grouping classes
3. Feature Selection:
   1. Datetime
4. Label/Target:
   1. hourly energy consumption
5. Split the data
   1. Train: 0.8 --> what about a validation data set ([K-fold Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html))
   2. Testing: 0.2
6. Choose Model (based of input features & labels/targets)
   1. Linear/Logistic Regression Model (Numeric/Classfication) implemented from scratch
   2. Other Model (Decision Trees, Forest model, Support Vector Regression, Neural Network, or others)

## 3. Train Model
1. Train, Evaluate, and Fine Tune
   1. Gradient Descent
   2. Gini Score 
## 4. Test Model
- redo training if accuracy is too low
  - decide on a threshold before hand

# Submission
- written project report
- presentation
- might be good:
  - jupyter notebook
