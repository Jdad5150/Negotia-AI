from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import joblib
import pandas as pd


model = joblib.load('./shared/salary_prediction_model.pkl')

app = Flask(__name__)

@app.route('/get-jobs')
def get_jobs():
    try:        
        return send_file('../shared/title_encoding.json')
    except Exception as e:
        return jsonify({'Error loading jobtitles': str(e)})    


@app.route('/get-exp-level')
def get_exp_level():
    try:
        return send_file('../shared/experience_encoding.json')
    except Exception as e:
        return jsonify({'Error loading experience levels': str(e)})


@app.route('/get-states')
def get_states():
    try:
        return send_file('../shared/state_encoding.json')
    except Exception as e:
        return jsonify({'Error loading states': str(e)})
    



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = request.json['features']

        features_ = {
            'state': features[0],
            'title': features[1],
            'experience': features[2]
        }

        features_df = pd.DataFrame([features_])
        
        prediction = model.predict(features_df)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'Prediction Error': str(e)})
    


if __name__ == '__main__':
    app.run(debug=True)