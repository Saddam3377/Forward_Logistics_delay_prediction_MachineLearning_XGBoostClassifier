# 🚚Forward_Logistics_delay_prediction_MachineLearning_XGBoostClassifier-
Leveraging Machine Learning to eliminate supply chain Logistics bottlenecks. This project deploys a XGBoost classifier Regression model to predict potential delivery delays between fixed service centers. By analyzing historical dispatch patterns, the system provides actionable insights to optimize forward logistics and improve delivery reliability.

Raw data contains 6 months of forward dispatch ( WH to Service center(SC), respective location) by using different courier modes like Air and 
Surface and by divergent onboard logistics partners.

This project builds a machine learning model to predict whether a courier shipment will be delayed using historical logistics data.
The model is trained using XGBoost and evaluated with advanced classification metrics, including Accuracy_Score, ROC-AUC, Precision, Recall, and Confusion Matrix.

**Dataset Features** 📇

Dispatch Warehouse,	SC Code, SC Status, Destination Pincode, Courier Assign,	Dispatch Dt, Mode, Avg_Transit_time, 	Courier delay AT_Sc_code, Courier delay AT_Pincode,	Rolling 7D_delay, Rolling 30D_delay, Target

🎯 **Objectives**

✔ Predict delivery delay probability

✔ Perform data preprocessing & feature engineering

✔ Train and evaluate XGBoost classifier

✔ Analyze model performance

**Python Key Libraries** 🔗

Pandas (Data Manipulation)

NumPy (Numerical Computing)

Scikit-learn (Model Training)

joblib (Model deploy)

📊 **Machine Learning Workflow**

Data Collection
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Model Training (XGBoost)
      │
      ▼
Model Evaluation
      │
      ▼
Prediction


🤖 **Model Used**

Model: XGBoost Classifier

Target Variable: Delay (0 = No Delay, 1 = Delay)


🎯 **Business Impact**

- Optimize courier performance by focusing on upcoming delayed shipments
  
- Reduce operational penalties

- Notify customers proactively

- Notify the relevant teams to adjust manpower accordingly to control costs and improve overall business profitability

📈 **Model Performance**

Accuracy: 91.77%

ROC-AUC Score: 90.15%

Classification Report:

Precision (Delay): 0.54, Recall (Delay): 0.90, F1-Score (Delay): 0.59

Confusion matrix heatmap :

<img width="565" height="415" alt="image" src="https://github.com/user-attachments/assets/95b1b4ab-3b4e-4357-8046-c7b372318606" />


ROC-AUC curve: To check how well the classifier separates two classes

<img width="565" height="415" alt="image" src="https://github.com/user-attachments/assets/fb476610-78e3-448b-98b2-bf9f23d7d2d4" />





