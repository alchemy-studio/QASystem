#!/usr/bin/python3

import json;
import requests;

def main(host):

    assert type(host) is str;
    uri = ''
    while True:
        line = input("question?>");
        if line == 'quit': break;
        data = json.dumps({'query': line});
        json_response = requests.post(host + '/qasystem', data = data, headers = {'content-type': 'application/json'});
        answers = json.loads(json_response.text)['answers'];
        for answer,score in answers.items():
            print('(' + str(score) + '):' + answer);

if __name__ == "__main__":

    main('http://127.0.0.1:5000');
