from flask import Flask, render_template, request, jsonify
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import pipeline

app = Flask(__name__)

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    context = request.form['context']
    question = request.form['question']
    
    if context.strip() == "" or question.strip() == "":
        return jsonify({'error': 'Context or question cannot be empty'}), 400
    
    result = qa_pipeline(question=question, context=context)
    return jsonify({'answer': result['answer']})

if __name__ == "__main__":
    app.run(debug=True)
