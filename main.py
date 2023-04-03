from flask import Flask, render_template, request
from transformers import pipeline

from modelo.trainer import train_model


app = Flask(__name__)
qa_pipeline = None

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/train')
def train():
    train_model()
    global qa_pipeline
    qa_pipeline = pipeline('question-answering', model='./model')

@app.route('/predict', methods=['POST'])
def predict():
    if qa_pipeline is None:
        return 'El modelo no ha sido entrenado todavía.'
    text = request.form['text']
    question = request.form['question']
    result = qa_pipeline(question=question, context=text)
    return result['answer']

if __name__ == '__main__':
    app.run(debug=True)
