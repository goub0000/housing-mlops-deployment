import gradio as gr
import pandas as pd
import joblib
import os

# Load the pre-trained model
model = joblib.load('model.pkl')

# Define feature names
feature_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                'guestroom', 'basement', 'hotwaterheating', 
                'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

# Define categorical features for conversion
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']

# Function to map yes/no to 1/0
def convert_yes_no(value):
    if value.lower() == 'yes':
        return 1
    else:
        return 0

# Function to predict house price
def predict_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                 basement, hotwaterheating, airconditioning, parking, 
                 prefarea, furnishingstatus):
    
    # Create a dataframe with the inputs
    input_data = {
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [convert_yes_no(mainroad)],
        'guestroom': [convert_yes_no(guestroom)],
        'basement': [convert_yes_no(basement)],
        'hotwaterheating': [convert_yes_no(hotwaterheating)],
        'airconditioning': [convert_yes_no(airconditioning)],
        'parking': [parking],
        'prefarea': [convert_yes_no(prefarea)]
    }
    
    # Handle furnishingstatus differently as it has three categories
    if furnishingstatus == "furnished":
        input_data['furnishingstatus_furnished'] = [1]
        input_data['furnishingstatus_semi-furnished'] = [0]
        input_data['furnishingstatus_unfurnished'] = [0]
    elif furnishingstatus == "semi-furnished":
        input_data['furnishingstatus_furnished'] = [0]
        input_data['furnishingstatus_semi-furnished'] = [1]
        input_data['furnishingstatus_unfurnished'] = [0]
    else:  # unfurnished
        input_data['furnishingstatus_furnished'] = [0]
        input_data['furnishingstatus_semi-furnished'] = [0]
        input_data['furnishingstatus_unfurnished'] = [1]
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Return the predicted price (formatted with comma separators)
    return f"₹{prediction:,.2f}"

# Create Gradio interface
with gr.Blocks(title="House Price Predictor") as demo:
    gr.Markdown("# House Price Prediction App")
    gr.Markdown("Enter the details of the house to get the predicted price.")
    
    with gr.Row():
        with gr.Column():
            area = gr.Number(label="Area (sq ft)", value=1000)
            bedrooms = gr.Slider(minimum=1, maximum=6, step=1, label="Bedrooms", value=2)
            bathrooms = gr.Slider(minimum=1, maximum=4, step=1, label="Bathrooms", value=1)
            stories = gr.Slider(minimum=1, maximum=4, step=1, label="Stories", value=1)
            parking = gr.Slider(minimum=0, maximum=3, step=1, label="Parking Spaces", value=0)
        
        with gr.Column():
            mainroad = gr.Radio(["Yes", "No"], label="Main Road Access", value="Yes")
            guestroom = gr.Radio(["Yes", "No"], label="Guest Room", value="No")
            basement = gr.Radio(["Yes", "No"], label="Basement", value="No")
            hotwaterheating = gr.Radio(["Yes", "No"], label="Hot Water Heating", value="No")
            airconditioning = gr.Radio(["Yes", "No"], label="Air Conditioning", value="No")
            prefarea = gr.Radio(["Yes", "No"], label="Preferred Area", value="No")
            furnishingstatus = gr.Radio(["furnished", "semi-furnished", "unfurnished"], 
                                      label="Furnishing Status", value="unfurnished")
    
    predict_btn = gr.Button("Predict Price")
    output = gr.Textbox(label="Predicted Price")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                basement, hotwaterheating, airconditioning, parking, 
                prefarea, furnishingstatus],
        outputs=output
    )
    
    gr.Markdown("### Note")
    gr.Markdown("This app uses a machine learning model trained on housing data. The predictions are estimates and should be used for reference only.")

# Launch the app
if __name__ == "__main__":
    demo.launch()
