CXX = /opt/homebrew/bin/g++-15
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

LIBS = -framework OpenGL -framework GLUT -lm -fopenmp

TARGET = flusim
SOURCE = main.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

# Use sudo make install since /usr/local/bin usually requires root permissions
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: clean install

