#ifndef SEARCH_ENGINE_H
#define SEARCH_ENGINE_H

#include <string>
#include <fstream>
#include <map>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/serialization/string.hpp>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include "cppjieba/Jieba.hpp"

using namespace std;
using namespace boost;
using namespace boost::multi_index;
using namespace boost::serialization;
using namespace boost::python;
using namespace cppjieba;

class SearchEngine {
 public:
  SearchEngine(string dict_root, string file = "");
  SearchEngine(const SearchEngine& sg);
  SearchEngine& operator=(const SearchEngine& sg);
  virtual ~SearchEngine() = default;

  dict query(string question, unsigned int count = 10);  
  bool loadFromTxt(string file);
  bool load(string file);
  bool save(string file);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & qa_list_;
    ar & term_list_;
    ar & term_frequency_list_;
    ar & doc_frequency_list_;
    ar & next_query_index_;
    ar & next_term_index_;
    ar & average_question_length_;
  }

 public:
  struct QA {
    unsigned int query_index_;
    string query_;
    string answer_;
  };
  struct Term {
    unsigned int term_index_;
    string term_;
  };
  struct TermFrequency {
    unsigned int query_index_;
    unsigned int term_index_;
    unsigned int term_frequency_;
  };
  struct DocFrequency {
    unsigned int term_index_;
    unsigned int doc_frequency_;
  };
  struct IndexByQueryId {};
  struct IndexByQuery {};
  struct IndexByTermId {};
  struct IndexByTerm {};
  struct IndexByQueryIdAndTermId {};
  using QAList = multi_index_container<
    QA,
    indexed_by<
      ordered_unique<
        boost::multi_index::tag<IndexByQueryId>,
        member<QA, unsigned int, &QA::query_index_>
      >,  // index by query index
      ordered_non_unique<
        boost::multi_index::tag<IndexByQuery>,
        member<QA, string, &QA::query_>
      >  // index by query string
    >
  >;
  using TermList = multi_index_container<
    Term,
    indexed_by<
      ordered_unique<
        boost::multi_index::tag<IndexByTermId>,
        member<Term, unsigned int, &Term::term_index_>
      >,  // index by term index
      ordered_unique<
        boost::multi_index::tag<IndexByTerm>,
        member<Term, string, &Term::term_>
      >  // index by term string
    >
  >;
  using TermFrequencyList = multi_index_container<
    TermFrequency,
    indexed_by<
      ordered_non_unique<
        boost::multi_index::tag<IndexByQueryId>,
        member<TermFrequency, unsigned int, &TermFrequency::query_index_>
      >,  // index by query index
      ordered_unique<
        boost::multi_index::tag<IndexByQueryIdAndTermId>,
        composite_key<
          TermFrequency,
          member<TermFrequency, unsigned int, &TermFrequency::query_index_>,
          member<TermFrequency, unsigned int, &TermFrequency::term_index_>
        >
      >  // index by query index and term index together
    >
  >;
  using DocFrequencyList = multi_index_container<
    DocFrequency,
    indexed_by<
      ordered_unique<
        boost::multi_index::tag<IndexByTermId>,
        member<DocFrequency, unsigned int, &DocFrequency::term_index_>
      >  // index by term index
    >
  >;
 private:
  float bm25(string question, unsigned int query_index);
  
  Jieba tokenizer_;
  QAList qa_list_;
  TermList term_list_;
  TermFrequencyList term_frequency_list_;
  DocFrequencyList doc_frequency_list_;
  unsigned int next_query_index_;
  unsigned int next_term_index_;
  unsigned int average_question_length_;
};

namespace boost {
namespace serialization {
template<class Archive>
void serialize(Archive & ar, SearchEngine::QA & qa, const unsigned int version) {
  ar & qa.query_index_;
  ar & qa.query_;
  ar & qa.answer_;
}

template<class Archive>
void serialize(Archive & ar, SearchEngine::Term & term, const unsigned int version) {
  ar & term.term_index_;
  ar & term.term_;
}

template<class Archive>
void serialize(Archive & ar, SearchEngine::TermFrequency & term_frequency, const unsigned int version) {
  ar & term_frequency.query_index_;
  ar & term_frequency.term_index_;
  ar & term_frequency.term_frequency_;
}

template<class Archive>
void serialize(Archive & ar, SearchEngine::DocFrequency & doc_frequency, const unsigned int version) {
  ar & doc_frequency.term_index_;
  ar & doc_frequency.doc_frequency_;
}
}
}

BOOST_PYTHON_MODULE(search_engine)
{
  class_<SearchEngine>("SearchEngine", init<string, optional<string> >())
      .def("query", &SearchEngine::query)
      .def("loadFromTxt", &SearchEngine::loadFromTxt)
      .def("load", &SearchEngine::load)
      .def("save", &SearchEngine::save)
  ;
}

#endif
