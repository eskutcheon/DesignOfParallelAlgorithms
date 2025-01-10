
NVCC=nvcc -O3 -arch sm_35
#NVCC=nvcc -O3 

deviceinfo: deviceinfo.cu
	$(NVCC) -o deviceinfo deviceinfo.cu

pi: pi.cu
	$(NVCC) -o pi pi.cu

testit: test.cu
	$(NVCC) -o testit test.cu
