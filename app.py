import pandas as pd
from flask import Flask, jsonify, request
import joblib

# load model
model = joblib.load('model.pkl')
model_columns = joblib.load("model_columns.pkl")

# app
app = Flask(__name__)

# routes
@app.route('/leadprediction', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.get_dummies(pd.DataFrame.from_dict(data))
    data_df = data_df.reindex(columns=model_columns, fill_value=0)

    # predictions
    result = model.predict(data_df)
    
    if result == 1:
        result_text = 'Hot Lead'
        boolean_text = 'true'
    else:
        result_text = 'Cold Lead'
        boolean_text = 'false'

    # send back to browser
    output = {'Lead Type':result_text, 'Can be Converted': boolean_text}

    # return data
    return jsonify(output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
