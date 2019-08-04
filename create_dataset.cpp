#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>

// compile with following command
// g++ convert.cpp -lboost_filesystem -lboost_system -lboost_program_options -lboost_regex -o convert

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::filesystem;

int main(int argc, char ** argv)
{
  string input_file, train_output_dir;
  options_description desc;
  desc.add_options()
    ("help,h","print current message")
    ("input,i",value<string>(&input_file),"input file path")
    ("train_output,o",value<string>(&train_output_dir),"train_output directory path");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);
  if (vm.count("help")) {
    cout<<desc;
    return EXIT_SUCCESS;
  } else if (1 != vm.count("input") || 1 != vm.count("train_output")) {
    cerr<<"input and train_output must be specified!"<<endl;
    return EXIT_FAILURE;
  }
  if (false == exists(input_file)) {
    cerr<<"input file doesn't exists!"<<endl;
    return EXIT_FAILURE;
  }
  remove_all(train_output_dir);
  std::ifstream in(input_file);
  if (false == in.is_open()) {
    cerr<<"can't open input file"<<endl;
    return EXIT_FAILURE;
  }
  remove_all(train_output_dir);
  create_directory(train_output_dir);
  std::ofstream train_out((path(train_output_dir) / "train.tsv").string());
  std::ofstream test_out((path(train_output_dir) / "test.tsv").string());
  if (false == train_out.is_open()) {
    cerr<<"can't open train file"<<endl;
    return EXIT_FAILURE;
  }
  if (false == test_out.is_open()) {
    cerr<<"can't open test file"<<endl;
    return EXIT_FAILURE;
  }
  map<string,string> question_answer;
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
      question_answer.insert(make_pair(question,answer));
    }
  }
  for (auto& sample1: question_answer) {
    for (auto& sample2: question_answer) {
      if (sample1.first == sample2.first) {
	train_out << 1 << "\t" << sample1.first << "\t" <<  sample2.second << endl;
      } else {
	train_out << 0 << "\t" << sample1.first << "\t" << sample2.second << endl;
	test_out << 0 << "\t" << sample1.first << "\t" << sample2.second << endl;
      }
    }
  }
  train_out.close();
  test_out.close();

  return EXIT_SUCCESS;
}

