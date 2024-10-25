import pandas as pd # Dataset manipulation
import numpy as np # Dataset manipulation
import seaborn as sns #statistical data visualization
import matplotlib.pyplot as plt # Visualization
from sklearn.model_selection import train_test_split # Splitting data into training and test sets
from sklearn.metrics import accuracy_score # Measures the accuracy of the model
from sklearn import metrics # Scores the performance of the model
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
# Load the dataset
data = pd.read_csv(r"C:\Users\dell\Desktop\diabetes_prediction_dataset.csv")
data.info()
data.describe()
data.isnull().sum()
a=pd.DataFrame(data)
diab_distribution = data['diabetes'].value_counts()/len(data)
#distribution des diabetes in data
plt.pie(diab_distribution,labels = ['non-diabetic', 'diabetic'],autopct = '%1.1f%%')
plt.title("Distribution of diabetics in dataset")
plt.show()
# Plotting
plt.scatter(a['age'], a['diabetes'])
plt.title('Diabetes vs Age')
plt.xlabel('Age')
plt.ylabel('Diabetes')
plt.grid(True)
plt.show()

plt.scatter(a['bmi'], a['diabetes'])
plt.title('Diabetes vs bmi')
plt.xlabel('bmi')
plt.ylabel('Diabetes')
plt.grid(True)
plt.show()

# Group data by gender and for each group
grouped_data = a.groupby(['gender', 'diabetes']).size().unstack()
grouped_data.plot(kind='bar', stacked=True, color=['green', 'orange'])

# Plotting
plt.title('Diabetes Counts by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No Diabetes', 'Diabetes'])
plt.show()

plt.scatter(a['hypertension'], a['diabetes'])
plt.title('Diabetes vs hypertention')
plt.xlabel('hypertention')
plt.ylabel('Diabetes')
plt.grid(True)
plt.show()




plt.scatter(a['heart_disease'], a['diabetes'])
plt.title('Diabetes vs heart_disease')
plt.xlabel('heart_disease')
plt.ylabel('Diabetes')
plt.grid(True)
plt.show()


plt.scatter(a['HbA1c_level'], a['diabetes'])
plt.title('Diabetes vs HbA1c_level')
plt.xlabel('HbA1c_level')
plt.ylabel('Diabetes')
plt.grid(True)
plt.show()



plt.scatter(a['blood_glucose_level'], a['diabetes'])
plt.title('Diabetes vs blood_glucose_level')
plt.xlabel('blood_glucose_level')
plt.ylabel('Diabetes')
plt.grid(True)
plt.show()



# Create a bar plot
grouped_data = a.groupby(['smoking_history', 'diabetes']).size().unstack().fillna(0)
grouped_data.plot(kind='bar', stacked=True, color=['green', 'orange'])

# Plotting
plt.title('Diabetes Counts by Smoking History')
plt.xlabel('Smoking History')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No Diabetes', 'Diabetes'])
plt.show()