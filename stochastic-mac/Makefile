CPP = /usr/local/opt/llvm/bin/clang++
CPPFLAGS =-std=c++11 -I/usr/local/opt/llvm/include -fopenmp -O3
LDFLAGS = -L/usr/local/opt/llvm/lib

col:
	$(CPP) $(CPPFLAGS) -o pairrank code/pairrank.cpp 

run:
	./pairrank

clean:
	rm *.o[0-9]* *.e[0-9]* *.o pairrank 
