CXXFLAGS=-Icppjieba/include -Icppjieba/deps -Icppjieba/deps/gtest/include `pkg-config --cflags python3` -O2
LIBS=-lboost_filesystem -lboost_system -lboost_program_options -lboost_regex -lboost_serialization -lboost_python3 `pkg-config --libs python3`
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: create_dataset create_database QASystem

create_dataset: create_dataset.o
	$(CXX) $^ $(LIBS) -o ${@}

create_database: create_database.o
	$(CXX) $^ $(LIBS) -o ${@}

QASystem: QASystem.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) $(OBJS) create_dataset create_database QASystem
