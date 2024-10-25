from sklearn.ensemble import RandomForestClassifier
import pandas as pd  # Dataset manipulation
from sklearn.model_selection import train_test_split  # Splitting data into training and test sets
from sklearn.metrics import accuracy_score  # Measures the accuracy of the model



# Read the dataset
# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Perform one-hot encoding for categorical columns
data_encoded = pd.get_dummies(data, columns=['gender', 'smoking_history'])  # Add other categorical columns as needed

# Update X and y using the modified dataset
X = data_encoded.drop('diabetes', axis=1)
y = data_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
ypred = rf_model.predict(X_test)

# Print the accuracy of the model
print("Accuracy of model: ", accuracy_score(y_test, ypred))
end_time = time.time()




