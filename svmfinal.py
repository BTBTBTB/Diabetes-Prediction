import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r"C:\Users\basma\OneDrive\Bureau\diabetes_prediction_dataset.csv")

# Perform one-hot encoding for categorical columns
data_encoded = pd.get_dummies(data, columns=['gender', 'smoking_history'])  # Add other categorical columns as needed

# Update X and y using the modified dataset
X = data_encoded.drop('diabetes', axis=1)
y = data_encoded['diabetes']


# Split the dataset into training and testing sets, maintaining class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:',accuracy)

