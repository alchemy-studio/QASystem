#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <tuple>
#include <map>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <boost/program_options.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include "cppjieba/Jieba.hpp"

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::serialization;
using namespace boost::archive;
using namespace boost::filesystem;
using namespace boost::python;
using namespace cppjieba;

python::object import(const string& module, const string& path, python::object& globals);

int main(int argc, char ** argv) {
  string dict_path;
  string database_path;
  options_description desc;
  desc.add_options()
    ("help,h", "print current message")
    ("database,D", value<string>(&database_path)->default_value("database.dat"), "TF/DF database file")
    ("dict,d", value<string>(&dict_path)->default_value("cppjieba/dict"), "setup dictionary path");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);
  if (vm.count("help")) {
    cout<<desc;
    return EXIT_SUCCESS;
  } else if (1 != vm.count("database") || 1 != vm.count("dict")) {
    cerr<<"TF/DF database file and dictionary must be specified!"<<endl;
    return EXIT_FAILURE;
  }
  path dict_root(dict_path);
  if (false == exists(dict_root) || false == is_directory(dict_root)) {
    cerr << "invalid dictionary directory!" <<endl;
    return EXIT_FAILURE;
  }
  std::ifstream in(database_path);
  if (false == in.is_open()) {
    cerr<<"can't open database.data! please execute create_database to generate it!"<<endl;
    return EXIT_FAILURE;
  }
  // load TF/DF
  vector<std::tuple<string, string, map<string, unsigned int> > > tf;
  map<string, unsigned int> df;
  long long avg_length;
  text_iarchive ia(in);
  ia >> tf >> df >> avg_length;
  // load tokenizer
  Jieba tokenizer(
    (dict_root/"jieba.dict.utf8").string(),
    (dict_root/"hmm_model.utf8").string(),
    (dict_root/"user.dict.utf8").string(),
    (dict_root/"idf.utf8").string(),
    (dict_root/"stop_words.utf8").string()
  );
  // Refer to https://en.wikipedia.org/wiki/Okapi_BM25
  std::function<float(string, unsigned int)> bm25([&](string query, unsigned int index) -> float {
    if (index >= tf.size()) {
      throw logic_error("index out of bound!");
    }
    vector<string> words;
    tokenizer.CutForSearch(query, words);
    float score = 0.;
    for_each(words.begin(), words.end(), [&](const string& word) {
      score += log((tf.size() - df[word] + 0.5) / (df[word] + 0.5)) *
                 (
                   get<2>(tf[index])[word] * (1.2 + 1) / 
                   (
                     get<2>(tf[index])[word] +
                     1.2 * (
                       1 - 0.75 + 0.75 * get<0>(tf[index]).size() / avg_length
                     )
                   )
                 );
    });
    return score;
  });
  // query answer loop
  try {
    Py_Initialize();
    python::object main = python::import("__main__");
    python::object globals = main.attr("__dict__");
    python::object BERT_module = import("BERT", "BERT.py", globals);
    python::object Predictor_module = import("Predictor", "Predictor.py", globals);
    python::object Predictor = Predictor_module.attr("Predictor");
    python::object predictor = Predictor();
    string line;
    while (1) {
      cout<<"question?>";
      cin>>line;
      trim(line);
      if ("quit" == line) {
	cout<<"quiting..."<<endl;
	break;
      }
      // search engine search for items above threshold
      vector<std::tuple<string, float, float> > candidate_answers;
      for (int i = 0 ; i < tf.size() ; i++) {
	float match_score = bm25(line, i);
	python::tuple retval = python::extract<python::tuple>(predictor.attr("predict")(line,get<1>(tf[i])));
	float relevance_score = python::extract<float>(retval[1]);
	candidate_answers.push_back(std::make_tuple(get<1>(tf[i]), match_score, relevance_score));
      }
      // sort in descend order
      sort(candidate_answers.begin(),candidate_answers.end(),[](const auto& a, const auto& b) {return exp(std::get<1>(a)) + exp(std::get<2>(a)) > exp(std::get<1>(b)) + exp(std::get<2>(b));});
      // print answers
      for (int i = 0 ; i < 3 && i < candidate_answers.size() ; i++) {
	cout<<"("<<exp(std::get<1>(candidate_answers[i])) + exp(std::get<2>(candidate_answers[i]))<<")"<<std::get<0>(candidate_answers[i])<<endl;
      }
    }
  } catch (const python::error_already_set&) {
    cerr<<"Python error occurred: "<<endl;
    PyErr_Print();
    exit(1);
  }
  Py_Finalize();
  
  return EXIT_SUCCESS;
}

python::object import(const string& module, const string& path, python::object& globals) {
  python::dict locals;
  locals["module_name"] = module;
  locals["path"] = path;
  python::exec("import imp\n"
	       "new_module = imp.load_module(module_name, open(path), path, ('py', 'U', imp.PY_SOURCE))\n",
	       globals,
	       locals);
  return locals["new_module"];
}

namespace boost {
  namespace serialization {
    template<class Archive>
    void serialize(Archive & ar, std::tuple<string, string, map<string, unsigned int> >& t, const unsigned int version) {
      ar & get<0>(t);
      ar & get<1>(t);
      ar & get<2>(t);
    }
  }
}
