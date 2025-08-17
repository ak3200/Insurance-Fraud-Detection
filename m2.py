import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


data = pd.read_csv('balanced_insurance_claims.csv')

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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_model = DecisionTreeClassifier(max_depth=1) 
model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))


if len(y_test.unique()) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]  
    auc = roc_auc_score(y_test, y_prob)
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
