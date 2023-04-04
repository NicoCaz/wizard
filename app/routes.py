#Contiene las rutas de la app web

from flask import render_template, request, jsonify
from app import app
from utils import qa_pipeline


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_question = request.form['input_question']
        prediction = qa_pipeline.predict(input_text, input_question)
        return jsonify({'prediction': prediction})
    else:
        return render_template('predict.html')
