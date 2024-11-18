CXX = g++

CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4`

LDFLAGS = `pkg-config --libs opencv4`


TARGET = SS

SRCS = main.cpp FeatureExtractor.cpp Construct_Graph.cpp EdmondsKarp.cpp


OBJS = $(SRCS:.cpp=.o)


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)