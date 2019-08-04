CXXFLAGS=-Icppjieba/include -Icppjieba/deps -Icppjieba/deps/gtest/include
LIBS=-lboost_filesystem -lboost_system -lboost_program_options -lboost_regex
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: convert create_index

convert: convert.o
	$(CXX) $^ $(LIBS) -o ${@}

create_index: create_index.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) $(OBJS) convert create_index
