import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Import dataset
leads_dataset = pd.read_csv('Leads.csv')
leads_dataset = leads_dataset.dropna()

# Create data pre-processing steps before plugging into model
leads_categorical_columns = ['Lead Origin',
                             'Lead Source',
                             'Last Activity',
                             'Lead Quality',
                             'Last Notable Activity']

leads_numeric_columns = ['TotalVisits',
                         'Total Time Spent on Website']

leads_response_columns = ['ToBeConverted']

lead_number = ['Lead Number']

dummy1 = pd.get_dummies(leads_dataset[leads_categorical_columns], drop_first=True)

# Adding the results to the master dataframe
leads_dataset = pd.concat([leads_dataset, dummy1], axis=1)

leads_dataset = leads_dataset.drop(leads_categorical_columns,axis=1)

# Putting feature variable to X
X = leads_dataset.drop(lead_number+leads_response_columns, axis=1)

# Putting response variable to y
y = leads_dataset[leads_response_columns]

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

scaler = StandardScaler()
X_train[leads_numeric_columns] = scaler.fit_transform(X_train[leads_numeric_columns])

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())
print(model.score(X_train, y_train.values.ravel()))
# Saving model to disk
joblib.dump(model, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
model = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")