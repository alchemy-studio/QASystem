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
  string input_file, output_file;
  options_description desc;
  desc.add_options()
    ("help,h","print current message")
    ("input,i",value<string>(&input_file),"input file path")
    ("output,o",value<string>(&output_file),"output file path");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);
  if (vm.count("help")) {
    cout<<desc;
    return EXIT_SUCCESS;
  } else if (1 != vm.count("input") || 1 != vm.count("output")) {
    cerr<<"input and output must be specified!"<<endl;
    return EXIT_FAILURE;
  }
  if (false == exists(input_file)) {
    cerr<<"input file doesn't exists!"<<endl;
    return EXIT_FAILURE;
  }
  remove_all(output_file);
  std::ifstream in(input_file);
  if (false == in.is_open()) {
    cerr<<"can't open input file"<<endl;
    return EXIT_FAILURE;
  }
  std::ofstream out(output_file);
  if (false == out.is_open()) {
    cerr<<"can't open open file"<<endl;
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
	out << 1 << "\t" << sample1.first << "\t" <<  sample2.second << endl;
      } else {
	out << 0 << "\t" << sample1.first << "\t" << sample2.second << endl;
      }
    }
  }
  out.close();
  return EXIT_SUCCESS;
}

