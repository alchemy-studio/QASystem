#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include "cppjieba/Jieba.hpp"

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace cppjieba;

int main(int argc, char ** argv) {
  string dict_path;
  string input_file;
  options_description desc;
  desc.add_options()
    ("help,h", "print current message")
    ("qa,q", value<string>(&input_file)->default_value("question_answer.txt"), "question and answer file")
    ("dict,d", value<string>(&dict_path)->default_value("cppjieba/dict"), "setup dictionary path");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);
  if (vm.count("help")) {
    cout<<desc;
    return EXIT_SUCCESS;
  } else if (1 != vm.count("qa") || 1 != vm.count("dict")) {
    cerr<<"question/answer file and dictionary must be specified!"<<endl;
    return EXIT_FAILURE;
  }
  if (false == exists(input_file)) {
    cerr<<"question/answer file doesn't exists!"<<endl;
    return EXIT_FAILURE;
  }
  path dict_root(dict_path);
  if (false == exists(dict_root) || false == is_directory(dict_root)) {
    cerr << "invalid dictionary directory!" <<endl;
    return EXIT_FAILURE;
  }
  Jieba jiaba(
    (dict_root/"jieba.dict.utf8").string(),
    (dict_root/"hmm_model.utf8").string(),
    (dict_root/"user.dict.utf8").string(),
    (dict_root/"idf.utf8").string(),
    (dict_root/"stop_words.utf8").string()
  );
  regex sample("(.+)\\t(.+)");
  while(false == in.eof()) {
    string line;
    getline(in,line);
    trim(line);
    if (line != "") {
      match_results<string::const_iterator> what;
      if (false == regex_match(line, what, sample)) {
	throw logic_error("invalid line format for sample!");
      }
      string question = string(what[1].begin(), what[1].end());
      string answer = string(what[2].begin(), what[2].end());
      trim(question);
      trim(answer);
      // tokenize the question
      vector<string> words;
      jieba.CutForSearch(question, words);
    }
  }
  return EXIT_SUCCESS;
}

