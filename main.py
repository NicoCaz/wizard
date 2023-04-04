from flask import Flask, render_template, request, jsonify
from app.utils import qa_pipeline
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')


app = Flask(__name__,template_folder='C:/Users/Nicolas/Desktop/wizard/app/templates')


@app.route('/')
def home():
    global qa_pipeline
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_question = request.form['input_question']
        prediction = qa_pipeline(input_text, input_question)
        tokens = tokenizer.encode(input_text)
        num_tokens = len(tokens)

        return jsonify({'prediction': prediction,'tokens':num_tokens})
    else:
        return render_template('predict.html')



if __name__ == '__main__':
    app.run(debug=True)
