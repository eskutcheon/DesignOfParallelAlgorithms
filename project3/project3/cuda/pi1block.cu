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

const int NBLOCKS = 1 ;
const int NTHREADS = 256 ;
__global__
void pikernel(int N, double *result) {
  double sum = 0 ;
  double dx = 1./double(N) ;
  int t = threadIdx.x ;
  int nseg = N/(NBLOCKS*NTHREADS) ;
  for(int i=t*nseg;i<(t+1)*nseg;++i) {
    double x = dx*(i+0.5) ;
    double f = sqrt(1.-x*x) ;
    sum += dx*f ;
  }
  //  __shared__ double scratch[NTHREADS] ;
  result[t] = 4*sum ;
  int offset = 512 ;
  while(offset > 1) {
    __syncthreads() ;
    if(t+offset < 1024)
      result[t] += result[t+offset] ;
    offset = offset / 2 ;
  }
  __syncthreads() ;
  if(t==0)
    *result = result[0]+result[1] ;
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
  cudaError_t err = cudaMalloc(&d_pi,NTHREADS*sizeof(double)) ;

  if(err!= cudaSuccess) {
    cerr << "cudaMalloc for d_pi failed" << endl ;
    cerr << "errror = " << cudaGetErrorString(err) << endl;
  }


  cudaEventRecord(start) ;
  // execute kernel with 1 thread block, 1 thread per block
  pikernel<<<NBLOCKS,NTHREADS>>>(N,d_pi) ;
  err = cudaGetLastError() ;
  if(err != cudaSuccess) {
    cerr << "kernal launch failed: " << cudaGetErrorString(err) << endl ;
  }
  cudaEventRecord(stop) ;
  double cudapi[1] ;
  err = cudaMemcpy(&cudapi[0],d_pi,sizeof(double),cudaMemcpyDeviceToHost) ;
  if(err!= cudaSuccess) {
    cerr << "cudaMemcpy for d_pi failed" << endl ;
    cerr << "error = " << cudaGetErrorString(err) << endl;
  }
  cudaEventSynchronize(stop) ;
  float milliseconds = 0 ;
  cudaEventElapsedTime(&milliseconds,start,stop) ;
  cout<< "pi kernel execution time = " << milliseconds << " ms" << endl ;

  double cudapisum = cudapi[0];
  computepi(N,&pi) ;
  cout << "pi = " << pi << " cudapi= " << cudapisum << endl ;
}

