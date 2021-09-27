CPP=g++
CPPFLAGS=-std=c++17 -march=native -I ./inc -O3 -lfftw3 -lfftw3l -lquadmath -lm -fopenmp 

LEMP:
	$(CPP) $(CPPFLAGS)  -c -fPIC src/LEMP.cpp -o out/LEMP.o  $(CPPFLAGS)
	$(CPP) $(CPPFLAGS) -shared -Wl,-soname,LEMP.so -o out/LEMP.so out/LEMP.o $(CPPFLAGS)

pop-dyn-planted:
	$(CPP) $(CPPFLAGS) -c -fPIC src/PopDynPlanted.cpp -o out/PDP.o $(CPPFLAGS)
	$(CPP) $(CPPFLAGS) -shared -Wl,-soname,PDP.so -o out/PDP.so  out/PDP.o $(CPPFLAGS)

fixed-cost:
	$(CPP) $(CPPFLAGS)  -c -fPIC src/fixed-cost.cpp -o out/fixedCost.o  $(CPPFLAGS)
	$(CPP) $(CPPFLAGS) -shared -Wl,-soname,fixedCost.so -o out/fixedCost.so out/fixedCost.o $(CPPFLAGS)

arb-kernel:
	$(CPP) $(CPPFLAGS)  -c -fPIC src/arbitrary-kernel.cpp -o out/arbKernel.o  $(CPPFLAGS)
	$(CPP) $(CPPFLAGS) -shared -Wl,-soname,arbKernel.so -o out/arbKernel.so  out/arbKernel.o $(CPPFLAGS)


