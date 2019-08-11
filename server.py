#!/usr/bin/python3

from flask import Flask, request;
from QASystem import QASystem;

app = Flask(__name__);
qasystem = QASystem();

@app.route('/')
def test():
    return 'QASystem server works!';

@app.route('/qasystem', method = ['POST'])
def query():
    params = request.get_json();
    question = params['query'];
    answer_score_list = qasystem.query(question,3);
    response = jsonify({'path': 'qasystem', 'query': params['query'], 'answers': answer_score_list});
    return response;
