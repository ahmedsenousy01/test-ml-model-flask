import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, make_response
app = Flask(__name__)


def load_pipeline(filename):
    try:
        return joblib.load(filename)
    except Exception:
        print("Could not load model")

model = load_pipeline("./pipeline.pkl")

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/api/predict", methods=["OPTIONS", "POST"])
def index():
    try:
        if request.method == "OPTIONS": # CORS preflight
            return _build_cors_preflight_response()
        elif request.method == "POST":
            data = request.get_json()    
            result = int(model.predict(pd.DataFrame(data, index=[0]))[0])
            return _corsify_actual_response(jsonify(status=True, data=result))
    except Exception as e:
        return _corsify_actual_response(jsonify(status=False, message=str(e)))

if __name__ == "__main__":
    app.run(debug=True, port=10000)
