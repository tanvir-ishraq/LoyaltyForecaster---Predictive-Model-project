# LoyaltyForecaster - Predictive Model project
End-to-end prediction model heavily using statistical data science covering the full machine learning pipleline. The project uses a lot of statistical data science techniques. The pipleline is procedurally documented with explanation step-by-step.

Hyperparameter-tuned Optimally engineered machine learning model that predicts customer loyalty; based on <br>
* Customer Account Information & Demographic <br>
* Services Information 
* Geographic Information <br>

To summary,
* Developed an end-to-end machine learning pipeline for predicting customer retention/churn.
* Heavily utilized statistical data science techniques to analyze and interpret complex datasets.
* Built a robust Model that effectively identifies customers likely to leave, data analysis enabling proactive customer retention strategies.

<h2 align="center">Data science (Preprocessing & Data Analysis)</h2>

### DATA EXPLORATION AND CLEANING:
**Method used**:  
* `EDA`(exploratory data analysis)
* Statistical `Median` <br><br>
**Key Results & derived Insights**: 
* Fixed Total Charges column feature’s datatype to Numeric.
* Replaced missing values under Total Charges column, by finding data statistical pattern of contract type feature. Replaced with Median on condition of contract type pattern
* Redundant repeated data are removed to reduce dimension complexity, such as 'Lat Long’, 'Country’, 'State’.

### FEATURE ENGINEERING AND SELECTION:
Created and transformed categorical and numerical features that could help in predicting customer churn.

#### Numerical Feature Engineering: 
**Method used**:  
* `Correlation Map Matrix(Pearson)` 
* `High Correlation Threshold` <br><br>
**Key Results & derived Insights**: 
* To reduce dimensions complexity by Pearson's Correlation Map Matrix with threshold value. 
* No Need to Drop Any Features for threshold value of 0.9.
* As the highest correlation coefficient of 0.9 is not matched, No Need to drop those numerical feature. 

**Method used**:  
* `Histogram`. *For* distribution analysis 
* `standard deviation`
* `outlier handling`<br><br>
**Key Results & derived Insights**: 
* The features mostly have normal distribution bell curve. 
* So, we can apply standard deviation to clean outliers. (2 x standard deviation)  
* Then replace outliers with median.

**Method used**:  
* `Min-Max Scaling` 
* `Data Normalization`<br><br>
**Key Results & derived Insights**: 
* This muti-dimension Dataset will require Scale Normalization. Applied MinMaxScaler() on Numerical feature. 

In machine learning, some feature’s min-max range differs from other features multiple times.  Data Normalization is a common practice which consists of transforming numeric columns to a common scale.

#### Categorical Feature Engineering: 
**Method used**:  
* `Bar Chart` with category vs Churn Relation
* `Visualization analysis for feature selection`<br><br>
**Key Results & derived Insights**: 
 * From the Chart visualization, The prominent features that correlate to churn are: ‘Partner’, ‘Dependents’, ‘Internet Service’, ‘Online Security’, ‘Online Backup’, ‘Device Protection’, ‘Tech Support’, ‘Streaming TV’, ‘Streaming Movies’, ‘Contract’, ‘Paperless Billing’, ‘Payment Method’.
Elaborately, We will see more of this Analysis in Bonus section.
* The relatively less impacting features are: ‘Gender’, ‘Senior Citizen’, ‘Phone Service’.

**Method used**:  
* `One Hot Encoding`<br><br>
**Key Results & derived Insights**: 
 * Dataset has 2-3 labels for each categorical columns. One Hot Encoding is perfect to equalize this duality relations. 
* Dataset is Encoded into numeric data for machine learning.

Labels are encoded in machine learning because most machine learning models can only operate on numerical data. Label encoding is a technique used to convert categorical variables into numerical format.

<h2 align="center">Machine Learning & Prediction</h2>


### DATASET PREPARATION:
**Method used**:  
* `Oversampling`. *For* class imbalance 
* `SMOTE` (Synthetic Minority Oversampling Technique) <br><br>
**Key Results & derived Insights**: 
* The churn yes data sample is very low which is crucial. This is not good for this binary classification.
 * SMOTE is applied for oversampling the minority class.

**Method used**:  
`stratify` (for balanced distribution in each dataset spilt) <br><br>
**Key Results & derived Insights**: 
* applied stratify for balanced distribution in train and test data. For correct evaluation.

 * Dataset split is made to be 75% train and 25% test data given 10,000 samples.


### MODEL SELECTION AND VALIDATION:
Engineered experiments to optimize machine learning model selection.

**Method used**:  
* Benchmark of Models 
* `random forests`, `logistic regression`, `SVM`, `KNN`, `gradient boosting` classification algorithms
* Monitor with baseline complexity of dataset <br><br>
**Key Results & derived Insights**: 
* Experimentation for model selection
* Selected algorithm: Random Forest (with 95% accuracy). 
    dataset baseline complexity was Accuracy: 50%

### HYPERPARAMETERS TUNING AND CROSS-VALIDATION:
Engineered experiments to optimize machine learning model selection.

**Method used**:  
* `Randomized Search` with `parameter grid`
* `5-fold cross-validation` 
*random forest parameters: `tree_depth`, `max_features selection`, `min_samples_split` <br><br>
**Key Results & derived Insights**: 
Best parameters combination with `100 iterations`: 
```
{
    'n_estimators': 50,  
    'min_samples_split': 2, 
    'max_features': 'log2’, 
    'max_depth': 35
}
```

### MODEL EVALUATION : 
#### CONFUSION MATRIX
 * We can see false negatives are much less. Only 81 compared to 1248. 
 * Minimal false negatives (i.e., correctly identifying all customers who are likely to churn) is very important for churn prediction. 
 * This is optimal performance. 

#### CLASSIFICATION REPORT, AUC-ROC
* Prediction task is focused on if user will churn/leave. so, minimizing false negatives(correctly identifying who will churn) is important. F1-score is a good indicat
* F1-score is 95% which is good.

* AUC score is 95% which is also good as it translates well to ROC curve. ROC diagram available in notebook

#### INSIGHT FROM CONFUSION MATRIX, CLASSIFICATION REPORT, AUC-ROC
* Minimizing false negatives (i.e., correctly identifying all customers who are likely to churn) may be more important than minimizing false positives. So, it could be more useful to look at F1 score than accuracy, for example. 

* An F1 score of 95% is a good score for the binary classification model.


### DEPLOYMENT PLAN:
Details provided in notebook. Outlined steps for deploying: 
* Model Serialization
* Deployment Approach
* Hosting Options

Monitoring and maintainance:
* Performance Monitoring
* Data Drift
* Retraining
* Versioning


<h2 align="center">Data Analysis Bonus</h2>
Details of the Data Analysis, visualization available in notebook. Extra data analysis yielded following insights: 

* We have an almost equal number of `men` and `women` customers in the data set.
* Both `genders` have similar churn rates. We see there is no difference in the churn rate.
* There are much fewer `senior` citizens customers.
* `Non-senior` citizens are the majority of customers which makes sense.
* Customers with `no partner` are a bit more likely to churn. 65% churn rate but major 45% don’t churn.
* Customers with `dependents` are substantially less prone to churn. 30% don’t churn. Only around 5% churn.
* The overall number of customers with `dependents` is lower than customers with `no dependents`.
* Customers with `no dependents` are a bit more likely to churn. 95% churn rate but major 70% don’t churn. Most customers have `dependents` which makes sense.
* Observing the churn rate, having `phone service` doesn’t contribute to churn.
* Having `multiple lines` doesn’t contribute to churn.
* The combination of `fiber optics` internet service is highly prone to churn. Improvements should be made or alternatives like `DSL` should be provided.
* The `DSL` internet service combination has a much lower rate of churn. The company can focus on `DSL` internet service.
* Customers who have `no internet service` when it comes to `streaming TV` or `streaming movies` are less likely to churn.
