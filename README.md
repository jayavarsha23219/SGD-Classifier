# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start

2. Import Necessary Libraries and Load Data

3. Split Dataset into Training and Testing Sets

4. Train the Model Using Stochastic Gradient Descent (SGD)

5. Make Predictions and Evaluate Accuracy

6. Generate Confusion Matrix

7. end

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: JAYAVARSHA T
RegisterNumber: 212223040075
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
iris = load_iris()
X = iris.data      # features: sepal length, sepal width, petal length, petal width
y = iris.target    # target: species (0=setosa, 1=versicolor, 2=virginica)

# Step 2: Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Define SGD Classifier
model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Predict for new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Example: sepal length, sepal width, petal length, petal width
prediction = model.predict(new_sample)
print("Predicted Species:", iris.target_names[prediction[0]])
```

## Output:
<img width="436" height="252" alt="image" src="https://github.com/user-attachments/assets/51d861e9-a28d-4bf5-adf6-493d5e46d95a" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
