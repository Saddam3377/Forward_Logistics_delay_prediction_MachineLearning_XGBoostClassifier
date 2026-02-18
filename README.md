# Forward_Logistics_delay_prediction_Machine-learning-Logistics-Regression-
Leveraging Machine Learning to eliminate supply chain Logistics bottlenecks. This project deploys a XGBoost classifier Regression model to predict potential delivery delays between fixed service centers. By analyzing historical dispatch patterns, the system provides actionable insights to optimize forward logistics and improve delivery reliability.

This project builds a machine learning model to predict whether a courier shipment will be delayed using historical logistics data.
The model is trained using XGBoost and evaluated with advanced classification metrics including Accuracy_Score , ROC-AUC, Precision, Recall, and Confusion Matrix.

**Dataset Features** 📇

Dispatch Warehouse,	SC Code, SC Status, Destination Pincode, Courier Assign,	Dispatch Dt, Mode, Avg_Transit_time, 	Courier delay AT_Sc_code, Courier delay AT_Pincode,	Rolling 7D_delay, Rolling 30D_delay, Target

**Python Key Libraries** 🔗

Pandas (Data Manipulation)

NumPy (Numerical Computing)

Scikit-learn (Model Training)

joblib (Model deploy)

🤖 **Model Used**

Model: XGBoost Classifier

Target Variable: Delay (0 = No Delay, 1 = Delay)

📈 **Model Performance**

Accuracy: 91.77%

ROC-AUC Score: 0.90

Classification Report:

Precision (Delay): 0.44, Recall (Delay): 0.89, F1-Score (Delay): 0.59

🎯 **Business Impact**

- Optimize courier performance by focusing upcoming delay shipments
  
- Reduce operational penalties

- Notify customers proactively








