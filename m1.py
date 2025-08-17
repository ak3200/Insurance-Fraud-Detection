import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

file_path = "balanced_insurance_claims.csv"  
data = pd.read_csv(file_path)

if 'incident_date' in data.columns:  
    data['incident_year'] = pd.to_datetime(data['incident_date']).dt.year
    data['incident_month'] = pd.to_datetime(data['incident_date']).dt.month
    data['incident_day'] = pd.to_datetime(data['incident_date']).dt.day
    data = data.drop(columns=['incident_date'])  


encoder = OneHotEncoder(sparse_output=False, drop='first') # one-hot encoding for categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
encoded_features = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))


numerical_columns = data.select_dtypes(include=['number']).columns # Combining with numerical features
data = pd.concat([data[numerical_columns], encoded_df], axis=1)

X = data.iloc[:, :-1]  
Y = data.iloc[:, -1]   

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)


print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print("F1-Score:", f1_score(Y_test, Y_pred))


if len(np.unique(Y_test)) == 2:
    Y_prob = rf_model.predict_proba(X_test)[:, 1]  
    auc = roc_auc_score(Y_test, Y_prob)
    
    
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()