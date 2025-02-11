from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from keras.models import load_model # type: ignore
import numpy as np
import json

model = load_model('./shared/demo_model.keras')

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
        return send_file('../shared/exp_level_encoding.json')
    except Exception as e:
        return jsonify({'Error loading experience levels': str(e)})


@app.route('/get-states')
def get_states():
    try:
        return send_file('../shared/state_encoding.json')
    except Exception as e:
        return jsonify({'Error loading states': str(e)})
    

@app.route('/get-worktypes')
def get_worktypes():
    try:
        return send_file('../shared/work_type_encoding.json')
    except Exception as e:
        return jsonify({'Error loading worktypes': str(e)})


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = request.json['features']
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})
    


if __name__ == '__main__':
    app.run(debug=True)