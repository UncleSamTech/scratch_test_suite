import os
from flask import Flask,request,jsonify
import pandas as pd
import joblib


#create FLASK APP

app = Flask(__name__)



@app.route('/predict',methods=['POST'])
def predict():

    #get json request
    feat_data = request.json
    #convert json to pandas df
    df = pd.DataFrame(feat_data)
    df = df.reindex(columns=col_names)

    #predict
    prediction = list(model.predict(df))

    #predict 
    return jsonify({'prediction':str(prediction)})

if __name__ == '__main___':
    model = joblib.load("finalmmodel.pkl") 
    tokenz  = joblib.load("tokenz.pkl")

    app.run(debug=True) 