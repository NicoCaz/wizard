#Contiene las rutas de la app web

from flask import render_template, request, jsonify
from app import app, qa_model


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/train')
def train():
    qa_model.train()
    return 'Training complete!'


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    input_question = request.form['input_question']
    prediction = qa_model.predict(input_text, input_question)
    return jsonify({'prediction': prediction})
