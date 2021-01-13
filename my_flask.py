import os
from flask import Flask, request, jsonify
import pickle
import sklearn
import pandas as pd
import json
import os

app = Flask(__name__)

testset_file = 'test_set.pkl'
model_file = 'linear_reg_model.pkl'
MODEL = pickle.load(open(model_file, 'rb'))
TEST_SET = pickle.load(open(testset_file, 'rb'))


@app.route('/')
def it_works():
    return 'It works!'


@app.route('/predict')
def predict():
    CRIM = request.args.get('CRIM')
    ZN = request.args.get('ZN')
    INDUS = request.args.get('INDUS')
    CHAS = request.args.get('CHAS')
    NOX = request.args.get('NOX')
    RM = request.args.get('RM')
    AGE = request.args.get('AGE')
    DIS = request.args.get('DIS')
    RAD = request.args.get('RAD')
    TAX = request.args.get('TAX')
    PTRATIO = request.args.get('PTRATIO')
    B = request.args.get('B')
    LSTAT = request.args.get('LSTAT')

    features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
    prediction = MODEL.predict([features])
    # http://127.0.0.1:5000/predict?CRIM=0.08&ZN=0.0&INDUS=4.08&CHAS=0.0&NOX=0.7&RM=5.55&AGE=70&DIS=2.9&RAD=5.0&TAX=300.0&PTRATIO=15.7&B=399&LSTAT=10.12

    return str(prediction)


@app.route('/<file>')
def house_price(file):
    try:
        with open(file) as f:
            DATA = json.load(f)

        features = [list(DATA[x].values()) for x in range(len(DATA))]
        prediction = MODEL.predict(features)
        result = []
        for i in range(len(prediction)):
            result.append({'prediction': prediction[i]})

        return jsonify(result)

    except FileNotFoundError:
        return "Enter valid JSON file in url."


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port), debug=True)
    else:
        app.run(debug=True)
