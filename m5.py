import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


data = pd.read_csv(r"balanced_insurance_claims.csv")


for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)
for col in data.select_dtypes(include=['number']).columns:
    data[col].fillna(data[col].median(), inplace=True)

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


target = 'fraud_reported'
X = data.drop(columns=[target])
y = data[target]


smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  # Input layer
    Dropout(0.25),  #dropout rate
    Dense(64, activation='relu'),  # hidden layer
    Dropout(0.25), 
    Dense(1, activation='sigmoid') # Output layer
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,  
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred = (model.predict(X_test) > 0.5).astype(int)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss During Training')
plt.xlabel('Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy During Training')
plt.xlabel('Epochs')

plt.show()
