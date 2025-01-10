# IMPORTANT
# Use "module load cuda/11.2.1" before compiling the code


NVCC=nvcc -O3 -arch sm_35


all: fluidcu fluid

fluidcu: fluid.cu
	$(NVCC) -o fluidcu fluid.cu

fluid: fluid.cc
	g++ -O3 -o fluid fluid.cc


clean:
	rm fluid fluidcu

distclean:
	rm -f fluid fluidcu *.out *~ *.dat
