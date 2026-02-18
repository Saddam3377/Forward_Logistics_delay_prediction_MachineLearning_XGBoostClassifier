## Forward Logistics_Delay_Prediction adn predictive deley date _ Machine learning model

#Reading the raw data 
import pandas as pd
df = pd.read_excel("C:\\Users\\Downloads\\New Folder\\Forward_train_rawData.xlsx")
df = df.sort_values(by= 'Dispatch Dt', ascending=True , )

#Extracting day, month and week from the dispatch date as model will not understand the date format and drop original Dispatch Dt column
df['Dispatch Dt'] = pd.to_datetime(df['Dispatch Dt'])
df['dispatch_month'] = df['Dispatch Dt'].dt.month
df['dispatch_week'] = df['Dispatch Dt'].dt.weekday
df['dispatch_day'] = df['Dispatch Dt'].dt.day
df = df.drop(columns=['Dispatch Dt'], axis = 1)

#Defining the categorical and numerical columns for preprocessing
cat_cols = ['Dispatch Wh', 'SC_code' , 'SC_status' , 'Dst Pincode','ODA','State','Region','Mode', 'Courier Align']
num_cols = ['dispatch_month', 'dispatch_week', 'dispatch_day', 'Avg_Transit_time', 'Courier delay_Sc_code', 'Courier delay_Pin' ,'Rolling 7D_delay', 'Rolling 30D_delay']

#Applying Meachine learning module to transform my raw data into binary format(0,1), so that my model can easily read
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


Preprocessing = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

x = df.drop(columns=['Target'], axis = 1)
y = df['Target']

## Applyung train_test Split method to divide ny raw transfrom data to train my model and test my model performance.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42 , stratify=y)

## Apply my machine learning LogisticRegression model 
# Pipeline = Pipeline(steps=[
#     ('Preprocessor', Preprocessing),
#     ('Classifier', LogisticRegression(C=0.5,class_weight = 'balanced' , solver = 'liblinear' , max_iter=2000))
# ])

# Pipeline.fit(x_train, y_train)

## Applying boosting method by different algotrith to improve my model performance
from xgboost import XGBClassifier
Pipeline_xgb = Pipeline(steps=[
    ('Preprocessor', Preprocessing), 
    ('Classifier', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42,scale_pos_weight=13, eval_metric='logloss'))
    ])

Pipeline_xgb.fit(x_train, y_train)

## checking my train model performance by using accuracy score, confusion matrix, classification report and roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# y_pred = Pipeline.predict(x_test)
y_pred = Pipeline_xgb.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


