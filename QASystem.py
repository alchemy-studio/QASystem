#!/usr/bin/python3

import operator;
from os.path import exists, join;
from search_engine import SearchEngine;
from Predictor import Predictor;

class QASystem(object):

    def __init__(self):

        if exists('database.dat'):
            # deserialize database is much faster.
            self.search_engine = SearchEngine('cppjieba/dict');
            print('deserialize the QA database...');
            self.search_engine.load('database.dat');
        else:
            # load database from txt is slower.
            print('load from QA database from txt format...');
            self.search_engine = SearchEngine('cppjieba/dict','question_answer.txt');
            self.search_engine.save('database.dat');
        self.predictor = Predictor();

    def query(self, question, count = 3):

        answer_scores = self.search_engine.query(question, count);
        answer_totalscores = dict();
        for answer, match in answer_scores.items():
            _, relevance = self.predictor.predict(question, answer);
            answer_totalscores[answer] = exp(match) + exp(relevance);
        # sort in descend order of total score
        descend = sorted(answer_totalscores, key = operator.itemgetter(1), reverse = True);
        return descend;

if __name__ == "__main__":

  qasystem = QASystem();
