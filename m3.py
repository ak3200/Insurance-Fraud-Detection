import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


data = pd.read_csv("balanced_insurance_claims.csv")

print(data.info())
print("Dataset shape:", data.shape)
print("Null values:\n", data.isnull().sum())


for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True) # Mode for categorical
for col in data.select_dtypes(include=['number']).columns:
    data[col].fillna(data[col].median(), inplace=True) # Median for numerical


if 'incident_date' in data.columns:  
    data['incident_year'] = pd.to_datetime(data['incident_date']).dt.year
    data['incident_month'] = pd.to_datetime(data['incident_date']).dt.month
    data['incident_day'] = pd.to_datetime(data['incident_date']).dt.day
    data = data.drop(columns=['incident_date'])  

    

encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_columns = data.select_dtypes(include=['object']).columns
encoded_features = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

numerical_columns = data.select_dtypes(include=['number']).columns
data = pd.concat([data[numerical_columns], encoded_df], axis=1)


iso_forest = IsolationForest(random_state=42, contamination=0.05)
data['anomaly'] = iso_forest.fit_predict(data.drop(columns='fraud_reported', errors='ignore'))
data = data[data['anomaly'] == 1].drop(columns='anomaly')


target = 'fraud_reported' 
X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_balanced, y_train_balanced)

y_pred = classifier.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')


if len(y_test.unique()) == 2:
    y_prob = classifier.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
