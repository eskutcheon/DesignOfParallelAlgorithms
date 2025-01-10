#include <iostream>
#include <cmath>

void computepi(int N, double *result) {
  double sum = 0 ;
  double dx = 1./double(N) ;
  for(int i=0;i<N;++i) {
    double x = dx*(i+0.5) ;
    double f = sqrt(1.-x*x) ;
    sum += dx*f ;
  }
  *result = 4.*sum ;
}

__global__
void pikernel(int N, double *result) {
  double sum = 0 ;
  double dx = 1./double(N) ;
  for(int i=0;i<N;++i) {
    double x = dx*(i+0.5) ;
    double f = sqrt(1.-x*x) ;
    sum += dx*f ;
  }
  *result = 4.*sum ;
}

int main(void) {
  using namespace std ;
  int N = 1<<20 ;
  cout.precision(16) ;
  cudaEvent_t start,stop ;
  cudaEventCreate(&start) ;
  cudaEventCreate(&stop) ;

  double pi = 0 ;
  double *d_pi ;
  cudaMalloc(&d_pi,sizeof(double)) ;



  cudaEventRecord(start) ;
  // execute kernel with 1 thread block, 1 thread per block
  pikernel<<<1,1>>>(N,d_pi) ;
  cudaEventRecord(stop) ;
  double cudapi[1] ;
  cudaMemcpy(&cudapi[0],d_pi,sizeof(double),cudaMemcpyDeviceToHost) ;
  cudaEventSynchronize(stop) ;
  float milliseconds = 0 ;
  cudaEventElapsedTime(&milliseconds,start,stop) ;
  cout<< "pi kernel execution time = " << milliseconds << " ms" << endl ;

  double cudapisum = cudapi[0];
  computepi(N,&pi) ;
  cout << "pi = " << pi << " cudapi= " << cudapisum << endl ;
}

