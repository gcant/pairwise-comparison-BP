CPP=g++
CPPFLAGS=-std=c++17 -march=native -I ./inc -O3 -lfftw3 -lfftw3l -lquadmath -lm -fopenmp 

lemp:
	$(CPP) $(CPPFLAGS)  -c -fPIC src/LEMP.cpp -o out/LEMP.o  $(CPPFLAGS)
	$(CPP) $(CPPFLAGS) -shared -Wl,-soname,LEMP.so -o out/LEMP.so out/LEMP.o $(CPPFLAGS)
