# Housing Price Prediction with MLOps

This repository demonstrates a simple MLOps implementation for a housing price prediction model. It uses a machine learning model to predict house prices based on various features and provides a user-friendly web interface for making predictions.

## Dataset

The model is trained on the `Housing.csv` dataset, which contains the following features:

- **price**: House price (target variable)
- **area**: House area in square feet
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **stories**: Number of stories
- **mainroad**: Whether the house is connected to the main road (Yes/No)
- **guestroom**: Whether the house has a guest room (Yes/No)
- **basement**: Whether the house has a basement (Yes/No)
- **hotwaterheating**: Whether the house has hot water heating (Yes/No)
- **airconditioning**: Whether the house has air conditioning (Yes/No)
- **parking**: Number of parking spaces
- **prefarea**: Whether the house is in a preferred area (Yes/No)
- **furnishingstatus**: Furnishing status (Furnished, Semi-Furnished, Unfurnished)

## Model Information

The application automatically trains a Linear Regression model on first run if no model file exists. The model is saved as `model.pkl` for future use to avoid retraining every time.

## Installation

Follow these steps to set up and run the application:

1. Clone the repository:
   ```
   git clone https://github.com/goub000/housing-mlops-deployment.git
   cd housing-mlops-deployment
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure the `Housing.csv` file is in the repository root directory.

## Usage

Run the application with:

```
python app.py
```

When you run the application for the first time, it will:
1. Check if a trained model exists
2. If not, automatically train a new model using the Housing.csv dataset
3. Save the model for future use
4. Start the Gradio web server (typically at http://127.0.0.1:7860/)

Open the provided URL in your web browser to access the application.

## How to Use the App

1. Fill in the house details in the form
2. Click the "Predict Price" button
3. View the predicted price

## MLOps Implementation

This project follows MLOps best practices:

1. **Version Control**: All code and data are tracked in Git
2. **Reproducibility**: Requirements are frozen in requirements.txt
3. **Automated Model Training**: The app automatically trains the model if needed
4. **Model Persistence**: Trained model is saved using joblib for reuse
5. **Web Deployment**: Easy deployment with Gradio
6. **Documentation**: Clear README with usage instructions
