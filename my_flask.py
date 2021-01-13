import os
from flask import Flask, request, jsonify
import pickle
import sklearn
import pandas as pd
import json
import os

app = Flask(__name__)

model_file = 'linear_reg_model.pkl'
MODEL = pickle.load(open(model_file, 'rb'))


@app.route('/')
def home():
    return '<h1>Boston Housing Price Prediction</h1>'


@app.route('/predict')
def prediction():
    try:
        df = pd.DataFrame.from_dict({feature: [float(request.args.get(feature))] for feature in request.args})
        prediction = MODEL.predict(df)[0]
    except:
        return "Invalid URL. \n Try: http://127.0.0.1:5000/prediction?CRIM=0.08&ZN=0.0&INDUS=4.08&CHAS=0.0&NOX=0.7&RM=5.55&AGE=70&DIS=2.9&RAD=5.0&TAX=300.0&PTRATIO=15.7&B=399&LSTAT=10.12"

    return str(prediction)


@app.route('/json', methods=['POST'])
def predict_many():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data)
        prediction = MODEL.predict(df)

    except:
        print("error")
        return jsonify({"Error": "invalid JSON file"})

    pred_lists = prediction.tolist()
    return json.dumps(pred_lists)


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port), debug=True)
    else:
        app.run(debug=True)
