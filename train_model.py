import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('Housing.csv')

# Convert categorical yes/no to 1/0
for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    data[col] = data[col].apply(lambda x: 1 if x == 'yes' else 0)

# Define features and target
X = data.drop('price', axis=1)
y = data['price']

# Create pipeline with preprocessing and model
categorical_features = ['furnishingstatus']
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline
pipeline.fit(X, y)

# Save the model
joblib.dump(pipeline, 'model.pkl')

print("Model trained and saved as model.pkl")
