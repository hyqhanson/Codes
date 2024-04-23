CXX = g++
CXXFLAGS = -std=c++17 -Wall 

SRCS = main.cc MultLinearReg.cc Logistic.cc Scaling.cc
OBJS = $(SRCS:.cc=.o)
EXEC = main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)