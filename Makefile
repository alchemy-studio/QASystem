CXXFLAGS= `pkg-config --cflags python3` -Icppjieba/include -Icppjieba/deps -Icppjieba/deps/gtest/include -fPIC -O2
LIBS=`pkg-config --libs python3` -lboost_python3 -lboost_filesystem -lboost_regex -lboost_serialization
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: create_dataset search_engine.so

create_dataset: create_dataset.o
	$(CXX) $^ $(LIBS) -lboost_system -lboost_program_options -o ${@}
	
search_engine.so: SearchEngine.o
	$(CXX) $^ $(LIBS) -shared -o ${@}

install: search_engine.so
	cp $^ ~/.local/lib/python3.6/site-packages/
	
clean:
	$(RM) $(OBJS) create_dataset search_engine.so
