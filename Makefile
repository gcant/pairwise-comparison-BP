CPP=g++
CPPFLAGS=-std=c++17 -march=native -I ./include -I ./ -O3 -lfftw3 -lfftw3l -lquadmath -lm -fopenmp 


lemp:
	$(CPP) $(CPPFLAGS)  -c -fPIC LEMP.cpp -o LEMP.o  $(CPPFLAGS)
	$(CPP) $(CPPFLAGS) -shared -Wl,-soname,LEMP.so -o LEMP.so LEMP.o $(CPPFLAGS)
