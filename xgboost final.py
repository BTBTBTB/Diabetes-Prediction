import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Perform one-hot encoding for categorical columns
data_encoded = pd.get_dummies(data, columns=['gender', 'smoking_history'])  # Add other categorical columns as needed

# Update X and y using the modified dataset
X = data_encoded.drop('diabetes', axis=1)
y = data_encoded['diabetes']

# Split the dataset into training and testing sets, maintaining class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = xgb_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print('XGBoost Accuracy:', accuracy)

# Allow the user to input their own information
user_input = {
    'gender': input("Enter gender (Male/Female/Other): "),
    'age': float(input("Enter age: ")),
    'hypertension': int(input("Is the user hypertensive? (0 for No, 1 for Yes): ")),
    'heart_disease': int(input("Does the user have heart disease? (0 for No, 1 for Yes): ")),
    'smoking_history': input("Enter smoking history (No Info/Current/Ever/Former/Never, Not Current): "),
    'bmi': float(input("Enter BMI: ")),
    'HbA1c_level': float(input("Enter HbA1c level:")),
    'blood_glucose_level': float(input("Enter blood glucose level: "))
}

# Create a DataFrame from user input
user_data = pd.DataFrame(user_input, index=[0])

# Perform one-hot encoding for user input
user_data_encoded = pd.get_dummies(user_data, columns=['gender', 'smoking_history'])

# Ensure that user_data_encoded has the same columns as X_train
user_data_encoded = user_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Use the trained Random Forest model to make a prediction
prediction =xgb_model.predict(user_data_encoded)

# Print the prediction
print("Predicted Diabetes Status:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")