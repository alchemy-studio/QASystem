#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/regex.hpp>
#include "SearchEngine.h"

using namespace std;
using namespace boost::archive;
using namespace boost::filesystem;

SearchEngine::SearchEngine(string dict_root, string file)
    : tokenizer_(
      (path(dict_root)/"jieba.dict.utf8").string(),
      (path(dict_root)/"hmm_model.utf8").string(),
      (path(dict_root)/"user.dict.utf8").string(),
      (path(dict_root)/"idf.utf8").string(),
      (path(dict_root)/"stop_words.utf8").string()),
      next_query_index_(0),
      next_term_index_(0) {
  if (file != "") {
    std::ifstream in(file, ios::binary);
    if (false == in.is_open()) {
      throw runtime_error("can't deserialize from file!");
    }
    binary_iarchive ia(in);
    ia >> *this;
  }
}

SearchEngine::SearchEngine(const SearchEngine& sg)
    : tokenizer_(sg.tokenizer_),
      qa_list_(sg.qa_list_),
      term_list_(sg.term_list_),
      term_frequency_list_(sg.term_frequency_list_),
      doc_frequency_list_(sg.doc_frequency_list_),
      next_query_index_(sg.next_query_index_),
      next_term_index_(sg.next_term_index_),
      average_question_length_(sg.average_question_length_) {
}

SearchEngine& SearchEngine::operator=(const SearchEngine& sg) {
  tokenizer_ = sg.tokenizer_;
  qa_list_ = sg.qa_list_;
  term_list_ = sg.term_list_;
  term_frequency_list_ = sg.term_frequency_list_;
  doc_frequency_list_ = sg.doc_frequency_list_;
  next_query_index_ = sg.next_query_index_;
  next_term_index_ = sg.next_term_index_;
  average_question_length_ = sg.average_question_length_;
  return *this;
}

dict SearchEngine::query(string question, unsigned int count) {
  vector<std::tuple<string, float> > scores;
  for (auto& qa : qa_list_) {
    float score = bm25(question, qa.query_index_);
    scores.push_back(std::make_tuple(qa.answer_, score));
  }
  // sort in descend order.
  sort(scores.begin(), scores.end(),
       [](const std::tuple<string, float>& a, const std::tuple<string, float>& b){
         return std::get<1>(a) > std::get<1>(b);    
       });
  if (scores.size() > count) scores.resize(count);
  dict answer_scores;
  for (auto& score: scores) {
    answer_scores[std::get<0>(score)] = std::get<1>(score);
  }
  return answer_scores;
}

float SearchEngine::bm25(string question, unsigned int query_index) {
  vector<string> words;
  tokenizer_.CutForSearch(question, words);
  float score = 0.;
  auto qa_iter = qa_list_.get<IndexByQueryId>().find(query_index);
  if (qa_iter == qa_list_.get<IndexByQueryId>().end()) {
    throw logic_error("can't find question/answer pair corresponding to query_index!");
  }
  for_each(words.begin(), words.end(), [&](const string& word) -> void {
    // 1) get term
    auto term_iter = term_list_.get<IndexByTerm>().find(word);
    // if it is not term being kept track of, skip it.
    if (term_iter == term_list_.get<IndexByTerm>().end()) return;
    // 2) get term frequency
    auto term_frequency_iter = term_frequency_list_.get<IndexByQueryIdAndTermId>().find(std::make_tuple(qa_iter->query_index_, term_iter->term_index_));
    // if the term is being kept track of, the term frequency must have it.
    if (term_frequency_iter == term_frequency_list_.get<IndexByQueryIdAndTermId>().end()) {
      throw logic_error("can't find term frequency of the given question!");
    }
    // 3) get doc frequency
    auto doc_frequency_iter = doc_frequency_list_.get<IndexByTermId>().find(term_iter->term_index_);
    // if the term is being kept track of, the doc frequency must have it.
    if (doc_frequency_iter == doc_frequency_list_.get<IndexByTermId>().end()) {
      throw logic_error("can't find doc frequency of the given term!");
    }
    score += log((qa_list_.size() - doc_frequency_iter->doc_frequency_ + 0.5) / (doc_frequency_iter->doc_frequency_ + 0.5)) *
                 (
                   term_frequency_iter->term_frequency_ * (1.2 + 1) / 
                   (
                     term_frequency_iter->term_frequency_ +
                     1.2 * (
                       1 - 0.75 + 0.75 * qa_iter->query_.size() / average_question_length_
                     )
                   )
                 );
  });
  return score;
}

bool SearchEngine::loadFromTxt(string file) {
  std::ifstream in(file);
  if (false == in.is_open()) {
    cerr<<"can't open txt file!"<<endl;
    return false;
  }
  
  regex sample("(.+)\\t(.+)");
  int length_sum = 0;
  while(false == in.eof()) {
    string line;
    getline(in,line);
    trim(line);
    if (line != "") {
      // for each non-empty querstion answer pair.
      match_results<string::const_iterator> what;
      if (false == regex_match(line, what, sample)) {
        cerr<<"invalid line format for sample!"<<endl;
        return false;
      }
      string question = string(what[1].begin(), what[1].end());
      string answer = string(what[2].begin(), what[2].end());
      trim(question);
      trim(answer);
      // 1) update QA list
      auto query_iter = qa_list_.get<IndexByQuery>().find(question);
      if (query_iter != qa_list_.get<IndexByQuery>().end()) {
        cerr<<"duplicate question! skip it!"<<endl;
        continue;
      } else {
        qa_list_.insert({next_query_index_++, question, answer});
        query_iter = qa_list_.get<IndexByQuery>().find(question);
      }
      // tokenize the question
      vector<string> words;
      tokenizer_.CutForSearch(question, words);
      // 2) update term list and term frequency list
      for (auto word: words) {
        // check whether word exists in term_list_.
        auto term_iter = term_list_.get<IndexByTerm>().find(word);
        if (term_iter == term_list_.get<IndexByTerm>().end()) {
          // add term if it doesn't exists.
          term_list_.insert({next_term_index_++, word});
          term_iter = term_list_.get<IndexByTerm>().find(word);
          assert(term_iter != term_list_.get<IndexByTerm>().end());
        }
        auto term_frequency_iter = term_frequency_list_.get<IndexByQueryIdAndTermId>().find(
            std::make_tuple(query_iter->query_index_, term_iter->term_index_));
        if (term_frequency_iter == term_frequency_list_.get<IndexByQueryIdAndTermId>().end()) {
          // add term frequency if it doesn't exists.
          term_frequency_list_.insert({query_iter->query_index_, term_iter->term_index_, 0});
          term_frequency_iter = term_frequency_list_.get<IndexByQueryIdAndTermId>().find(
              std::make_tuple(query_iter->query_index_, term_iter->term_index_));
        }
        term_frequency_list_.get<IndexByQueryIdAndTermId>().modify(term_frequency_iter,
                                                         [](TermFrequency & tf) {tf.term_frequency_++;});
      }
      // 3) update document frequency
      set<string> tokens(words.begin(), words.end());
      for (auto token: tokens) {
        auto term_iter = term_list_.get<IndexByTerm>().find(token);
        assert(term_iter != term_list_.get<IndexByTerm>().end());
        auto doc_frequency_iter = doc_frequency_list_.get<IndexByTermId>().find(term_iter->term_index_);
        if (doc_frequency_iter == doc_frequency_list_.get<IndexByTermId>().end()) {
            // add document frequency if it doesn't exists.
            doc_frequency_list_.insert({term_iter->term_index_, 0});
            doc_frequency_iter = doc_frequency_list_.get<IndexByTermId>().find(term_iter->term_index_);
        }
        doc_frequency_list_.get<IndexByTermId>().modify(doc_frequency_iter,
                                                        [](DocFrequency & df) {df.doc_frequency_++;});
      }
      // update average length
      length_sum += question.size();
    }  // if line is not empty
  }  // for each question answer pair
  // update average document length.
  average_question_length_ = length_sum * 1.0 / qa_list_.size();

  return true;
}

bool SearchEngine::load(string file) {
  std::ifstream in(file, ios::binary);
  if (false == in.is_open()) {
    cerr<<"can't deserialize from file!"<<endl;
    return false;
  }
  binary_iarchive ia(in);
  ia >> *this;
  return true;
}

bool SearchEngine::save(string file) {
  std::ofstream out(file, ios::binary);
  if (false == out.is_open()) {
    cerr<<"can't serialize to file!"<<endl;
    return false;
  }
  binary_oarchive oa(out);
  oa << *this;
  return true;
}
