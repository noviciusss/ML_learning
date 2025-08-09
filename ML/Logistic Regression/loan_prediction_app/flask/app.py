from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Get the absolute path to the directory where app.py is located
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths to the model and scaler files
MODEL_PATH = os.path.join(APP_ROOT, 'models', 'loan_prediction_model.pkl')
SCALER_PATH = os.path.join(APP_ROOT, 'models', 'scaler.pkl')

print("--- DEBUGGING FILE PATHS ---")
print(f"APP_ROOT: {APP_ROOT}")
print(f"Calculated MODEL_PATH: {MODEL_PATH}")
print(f"Calculated SCALER_PATH: {SCALER_PATH}")
print(f"Does MODEL_PATH exist? {os.path.exists(MODEL_PATH)}")
print(f"Does SCALER_PATH exist? {os.path.exists(SCALER_PATH)}")
print("--- END DEBUGGING FILE PATHS ---")

model = None
scaler = None

try:
    print("Attempting to load model and scaler using joblib...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully using joblib!")
except FileNotFoundError:
    print(f"Error: FileNotFoundError. One or both model files not found at the specified paths.")
    print(f"Checked for model at: {MODEL_PATH}")
    print(f"Checked for scaler at: {SCALER_PATH}")
    print("Please ensure the files 'loan_prediction_model.pkl' and 'scaler.pkl' exist in the 'flask/models/' directory.")
    model = None # Ensure they are None if loading fails
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print(f"Type of error: {type(e)}")
    print("This could be due to a version mismatch (scikit-learn, joblib, python) between saving and loading, or if the files are corrupted.")
    print("Ensure the files in the 'flask/models/' directory were saved using 'joblib.dump' with compatible library versions.")
    model = None # Ensure they are None if loading fails
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("--- PREDICT ROUTE ---")
    print(f"Request Method: {request.method}")
    print(f"Request Content-Type: {request.content_type}")
    print(f"Request Content-Length: {request.content_length}")
    
    raw_data_sample = request.get_data(as_text=False)[:200]
    try:
        print(f"Request Raw Data Sample (first 200 bytes as text, if possible): {raw_data_sample.decode('utf-8', errors='replace')}")
    except Exception:
        print(f"Request Raw Data Sample (first 200 bytes, non-decodable): {raw_data_sample}")

    if model is None or scaler is None:
        print("Error: Model or scaler is None. Cannot proceed with prediction.")
        return render_template('result.html', error='Model or scaler not loaded. Please check server logs.')
    
    data = {}
    try:
        if request.form:
            data = request.form.to_dict()
            print(f"Data successfully parsed from request.form: {data}")
        elif request.is_json:
            print("request.form was empty. Attempting to parse as JSON...")
            json_data = request.get_json()
            if json_data is not None:
                data = json_data
                print(f"Data successfully parsed from request.get_json(): {data}")
            else:
                print("request.is_json was true, but get_json() returned None or failed.")
        else:
            print("request.form is empty and request is not JSON. No standard form data or JSON detected.")

        if not data:
            print("Error: No parsable data (form or JSON) received in the request payload.")
            full_raw_data = request.get_data(as_text=True)
            print(f"Full Raw Request Body for debugging: {full_raw_data}")
            return render_template('result.html', error='No input data received by the server. Please check form submission.')
        
        print(f"Final data to be processed: {data}")
        
        # All features expected from the HTML form
        all_form_feature_names = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ]

        # Features that the model and scaler actually expect (after dropping correlated ones)
        # Assuming 'loan_amount' and 'luxury_assets_value' were the ones dropped.
        # Adjust this list based on the actual `corr_feat` from your notebook.
        feature_names_for_model = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            # 'loan_amount', # Dropped
            'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value',
            # 'luxury_assets_value', # Dropped
            'bank_asset_value'
        ]
        
        features_for_model_input = [] # This will be fed to the scaler/model
        all_input_values = {} # This will store all form inputs for display

        # Populate all_input_values and features_for_model_input
        for feature_name in all_form_feature_names:
            if feature_name not in data:
                print(f"Error: Missing feature '{feature_name}' in received data dict. Available keys: {list(data.keys())}")
                return render_template('result.html', error=f"Missing input field: {feature_name}")
            try:
                value_str = str(data[feature_name])
                value = float(value_str)
                all_input_values[feature_name] = value # Store all inputs for display

                if feature_name in feature_names_for_model: # Only add to model input if it's an expected feature
                    features_for_model_input.append(value)

            except ValueError:
                print(f"Error: Could not convert feature '{feature_name}' with value '{data.get(feature_name)}' to float.")
                return render_template('result.html', error=f"Invalid value for {feature_name}: '{data.get(feature_name)}'")
            except TypeError:
                print(f"Error: TypeError for feature '{feature_name}' with value '{data.get(feature_name)}'. Could not convert to string or float.")
                return render_template('result.html', error=f"Invalid type or missing value for {feature_name}: '{data.get(feature_name)}'")

        print(f"All input values (for display): {all_input_values}")
        print(f"Features for model input (after dropping correlated): {features_for_model_input}")

        if len(features_for_model_input) != len(feature_names_for_model):
            print(f"Error: Mismatch in expected model features. Expected {len(feature_names_for_model)}, got {len(features_for_model_input)}")
            return render_template('result.html', error='Internal server error: Feature mismatch for model.')

        features_array = np.array(features_for_model_input).reshape(1, -1)
        
        print("Scaling features for model...")
        standardized_features = scaler.transform(features_array)
        print(f"Standardized features for model: {standardized_features}")
        
        print("Making prediction...")
        prediction_val = model.predict(standardized_features)
        print(f"Raw prediction: {prediction_val}")
        
        prediction_proba_val = model.predict_proba(standardized_features)
        print(f"Prediction probability: {prediction_proba_val}")
        
        loan_status_str = "Approved" if prediction_val[0] == 1 else "Rejected"
        prob_approved_val = float(prediction_proba_val[0][1])
        prob_rejected_val = float(prediction_proba_val[0][0])
        
        result_dict = {
            'prediction': int(prediction_val[0]),
            'loan_status': loan_status_str,
            'prob_approved': prob_approved_val,
            'prob_rejected': prob_rejected_val,
            'input_features': all_input_values # Use all_input_values here for display
        }
        print(f"Result to be rendered: {result_dict}")
        return render_template('result.html', result=result_dict)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        print(traceback.format_exc()) 
        return render_template('result.html', error=f'An unexpected error occurred during prediction: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    print("--- API PREDICT ROUTE ---")
    if model is None or scaler is None:
        print("API Error: Model or scaler is None.")
        return jsonify({'error': 'Model or scaler not loaded'}), 500
    
    try:
        data = request.get_json()
        if data is None:
            print("API Error: No JSON data received or failed to parse.")
            return jsonify({'error': 'Request body must be JSON and parseable.'}), 400
            
        print(f"API Form data received: {data}")
        
        # Features that the model and scaler actually expect
        feature_names_for_model = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            # 'loan_amount', # Dropped
            'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value',
            # 'luxury_assets_value', # Dropped
            'bank_asset_value'
        ]
        
        features_for_model_input = []
        for feature_name in feature_names_for_model: # Iterate only over features needed for the model
            if feature_name not in data:
                print(f"API Error: Missing feature '{feature_name}' in JSON data.")
                return jsonify({'error': f"Missing input field for model: {feature_name}"}), 400
            try:
                features_for_model_input.append(float(str(data[feature_name])))
            except ValueError:
                print(f"API Error: Could not convert feature '{feature_name}' with value '{data.get(feature_name)}' to float.")
                return jsonify({'error': f"Invalid value for {feature_name}: '{data.get(feature_name)}'"}), 400
            except TypeError:
                print(f"API Error: TypeError for feature '{feature_name}' with value '{data.get(feature_name)}'.")
                return jsonify({'error': f"Invalid type for {feature_name}: '{data.get(feature_name)}'"}), 400

        if len(features_for_model_input) != len(feature_names_for_model):
            print(f"API Error: Mismatch in expected model features. Expected {len(feature_names_for_model)}, got {len(features_for_model_input)}")
            return jsonify({'error': 'Internal server error: Feature mismatch for model.'}), 500

        features_array = np.array(features_for_model_input).reshape(1, -1)
        standardized_features = scaler.transform(features_array)
        
        prediction = model.predict(standardized_features)
        prediction_proba = model.predict_proba(standardized_features)
        
        response_data = {
            'prediction': int(prediction[0]),
            'loan_status': "Approved" if prediction[0] == 1 else "Rejected",
            'probability': {
                'approved': float(prediction_proba[0][1]),
                'rejected': float(prediction_proba[0][0])
            }
            # Note: The API response doesn't typically include all raw inputs unless specified.
            # If you want to include them, you can add: 'input_features': data
        }
        print(f"API Result: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"API Error during prediction: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)