CXXFLAGS=-Icppjieba/include -Icppjieba/deps -Icppjieba/deps/gtest/include
LIBS=-lboost_filesystem -lboost_system -lboost_program_options -lboost_regex -lboost_serialization
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: convert create_database

convert: convert.o
	$(CXX) $^ $(LIBS) -o ${@}

create_database: create_database.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) $(OBJS) convert create_database

