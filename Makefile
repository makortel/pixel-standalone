CXXFLAGS := -O2 -std=c++17
CXX := g++

default: naive

naive:
	$(CXX) $(CXXFLAGS) -DDIGI_NAIVE -o main-naive main.cc

debug-naive:
	$(CXX) $(CXXFLAGS) -DDIGI_NAIVE -g -o main-naive main.cc
