import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
app = Flask(__name__)

test_data = np.array([[42,
                      0,
                      1506,
                      0,
                      50,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      0,
                      1,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      0,
                      1,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      0,
                      0,
                      0,
                      0,
                      1]]
                     )

df = pd.DataFrame(test_data, columns=['age',
                                      'education_num',
                                      'capital_gain',
                                      'capital_loss',
                                      'hours_per_week',
                                      'country_ Cambodia',
                                      'country_ Canada',
                                      'country_ China',
                                      'country_ Columbia',
                                      'country_ Cuba',
                                      'country_ Dominican-Republic',
                                      'country_ Ecuador',
                                      'country_ El-Salvador',
                                      'country_ England',
                                      'country_ France',
                                      'country_ Germany',
                                      'country_ Greece',
                                      'country_ Guatemala',
                                      'country_ Haiti',
                                      'country_ Holand-Netherlands',
                                      'country_ Honduras',
                                      'country_ Hong',
                                      'country_ Hungary',
                                      'country_ India',
                                      'country_ Iran',
                                      'country_ Ireland',
                                      'country_ Italy',
                                      'country_ Jamaica',
                                      'country_ Japan',
                                      'country_ Laos',
                                      'country_ Mexico',
                                      'country_ Nicaragua',
                                      'country_ Outlying-US(Guam-USVI-etc)',
                                      'country_ Peru',
                                      'country_ Philippines', 
                                      'country_ Poland',
                                      'country_ Portugal',
                                      'country_ Puerto-Rico',
                                      'country_ Scotland',
                                      'country_ South',
                                      'country_ Taiwan',
                                      'country_ Thailand',
                                      'country_ Trinadad&Tobago',
                                      'country_ United-States',
                                      'country_ Vietnam',
                                      'country_ Yugoslavia',
                                      'gender_ Female',
                                      'gender_ Male',
                                      'workclass_ Federal-gov',
                                      'workclass_ Local-gov',
                                      'workclass_ Private',
                                      'workclass_ Self-emp-inc',
                                      'workclass_ Self-emp-not-inc',
                                      'workclass_ State-gov',
                                      'workclass_ Without-pay',
                                      'occupation_ Adm-clerical',
                                      'occupation_ Armed-Forces',
                                      'occupation_ Craft-repair',
                                      'occupation_ Exec-managerial',
                                      'occupation_ Farming-fishing',
                                      'occupation_ Handlers-cleaners',
                                      'occupation_ Machine-op-inspct',
                                      'occupation_ Other-service',
                                      'occupation_ Priv-house-serv',
                                      'occupation_ Prof-specialty',
                                      'occupation_ Protective-serv',
                                      'occupation_ Sales',
                                      'occupation_ Tech-support',
                                      'occupation_ Transport-moving',
                                      'marital_status_ Divorced',
                                      'marital_status_ Married-AF-spouse',
                                      'marital_status_ Married-civ-spouse',
                                      'marital_status_ Married-spouse-absent',
                                      'marital_status_ Never-married',
                                      'marital_status_ Separated',
                                      'marital_status_ Widowed',
                                      'relationship_ Husband',
                                      'relationship_ Not-in-family',
                                      'relationship_ Other-relative',
                                      'relationship_ Own-child',
                                      'relationship_ Unmarried',
                                      'relationship_ Wife',
                                      'race_ Amer-Indian-Eskimo',
                                      'race_ Asian-Pac-Islander',
                                      'race_ Black',
                                      'race_ Other',
                                      'race_ White'])
# 'age','education_num','capital_gain','capital_loss','hours_per_week','country','gender','race','occupation','relationship','marital_status'
# workclass_, 

# step 1: user puts input in select boxes in frontend
# step 2: frontend sends the user input through http
# step 3: from that input generate a numpy array
# step 4: pass the data to the model and return it's prediction

def load_pipeline(filename):
    """
    Load the Gradient Boosting classifier model from a pickle file.

    Parameters:
    - filename (str): The path to the pickle file.

    Returns:
    - gb_classifier (GradientBoostingClassifier): The loaded Gradient Boosting classifier model.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_real_pipeline(filename):
    """
    Load the Gradient Boosting classifier model from a pickle file.

    Parameters:
    - filename (str): The path to the pickle file.

    Returns:
    - gb_classifier (GradientBoostingClassifier): The loaded Gradient Boosting classifier model.
    """
    try:
        return joblib.load(filename)
    except Exception:
        print("Could not load model")


# load the model and give it data
model = load_real_pipeline("./pipeline.pkl")

@app.get("/api/predict")
def index():
    result = int(model.predict(df)[0])
    return jsonify(status=True, data=result)

@app.post("/api/pass-data")
def pass_data():
    # get the user data and store it in a variable
    data = request.get_json()
    data_values = model.named_steps["scaler"].transform(np.array(list(data.values())))
    # print(model.named_steps)
    
    result = model.predict(data_values)
    print(result)
    return data;

if __name__ == "__main__":
    app.run(debug=True, port=10000)
