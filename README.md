# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the libraries and read the data frame using pandas.
Calculate the null values present in the dataset and apply label encoder.
Determine test and training data set and apply decison tree regression in dataset.
calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  
*/
# Import libraries

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load the data

df = pd.read_csv('Salary.csv')

# Display the data

print("Salary Data:")

print(df)
print()

# Prepare data

X = df[['Level']]  # Feature (Level)
y = df['Salary']    # Target (Salary)

# Create and train the model

model = DecisionTreeRegressor()
model.fit(X, y)

# Make predictions for all levels

predictions = model.predict(X)

# Show predictions

print("Actual vs Predicted Salaries:")
for i in range(len(df)):
    print(f"Level {df.iloc[i]['Level']}: Actual=${df.iloc[i]['Salary']}, Predicted=${int(predictions[i])}")

# Calculate accuracy (R² score)

accuracy = model.score(X, y)
print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")

# Predict salary for a new level

new_level = [[6.5]]
predicted_salary = model.predict(new_level)
print(f"\nPredicted Salary for Level 6.5: ${int(predicted_salary[0])}")

# Simple visualization

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=100)
plt.plot(X, predictions, color='red', label='Decision Tree Predictions', linewidth=2)
plt.xlabel('Level')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction using Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
<img width="440" height="598" alt="image" src="https://github.com/user-attachments/assets/c29fef01-4691-42fb-892f-9d1197307e55" />
<img width="882" height="570" alt="image" src="https://github.com/user-attachments/assets/6313b5bb-17b3-41aa-b90e-78b144d06187" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
