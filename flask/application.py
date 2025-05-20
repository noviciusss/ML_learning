from flask import Flask,request, jsonify,render_template
import pickle 
import numpy as np
import pandas as pd , matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

application = Flask(__name__) #creating a flask app
# Load the model
app = application

#render_template allows us to render HTML templates
ridge_model = pickle.load(open('flask/models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('flask/models/scaler.pkl', 'rb')) #loading ridge.pkl again!

@app.route('/') #default route
def index():
       return render_template('index.html')

@app.route("/predict",methods=['Get','POST']) #route for prediction\
def predict_data():
       if request.method =="POST":
              Temperature = float(request.form.get('temperature'))
              RH = float(request.form.get('rh'))
              Ws = float(request.form.get('ws'))
              Rain = float(request.form.get('rain'))
              FFMC = float(request.form.get('ffmc'))
              DMC = float(request.form.get('dmc'))
              ISI = float(request.form.get('isi'))
              Classes = float(request.form.get('classes'))
              Region = float(request.form.get('region'))    
              
              new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
              result = ridge_model.predict(new_data_scaled) 
              return render_template('home.html', results=result[0]) #rendering the home.html file with the prediction result
       else:
              return render_template('home.html')
if __name__ == '__main__':
       app.run(debug=True) #start the flask app
