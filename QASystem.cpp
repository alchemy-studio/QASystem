#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <boost/program_options.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::serialization;
using namespace boost::archive;

int main(int argc, char ** argv) {
  std::ifstream in("database.dat");
  if (false == in.is_open()) {
    cerr<<"can't open database.data! please execute create_database to generate it!"<<endl;
    return EXIT_FAILURE;
  }
  vector<std::tuple<string, string, map<string, unsigned int> > > tf;
  map<string, unsigned int> df;
  long long avg_length;
  text_iarchive ia(in);
  ia >> tf >> df >> avg_length;
  
  return EXIT_SUCCESS;
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
