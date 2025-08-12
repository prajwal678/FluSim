CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp
LIBS = -lglut -lGL -lGLU -lm -fopenmp
TARGET = flusim
SOURCE = main.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: clean install
