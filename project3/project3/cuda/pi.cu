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

const int NBLOCKS = 64 ;
const int NTHREADS = 1024 ;
__global__
void pikernel(int N, double *result) {
  double sum = 0 ;
  double dx = 1./double(N) ;
  int t = threadIdx.x ;
  int nseg = N/(NBLOCKS*NTHREADS) ;
  int b = blockIdx.x ;
  for(int i=t*nseg;i<(t+1)*nseg;++i) {
    double x = dx*(i+b*nseg*NTHREADS+0.5) ;
    double f = sqrt(1.-x*x) ;
    sum += dx*f ;
  }
  __shared__ double scratch[1024] ;
  scratch[t] = 4*sum ;
  int offset = NTHREADS ;
  while(offset > 2) {
    offset = offset/2 ;
    __syncthreads() ;
    if(t < offset)
      scratch[t] += scratch[t+offset] ;
  }
  __syncthreads() ;
  if(t==0)
    result[b] = scratch[0]+scratch[1] ;
}

__global__ 
void pisum(int N, double *result) {
  __shared__ double scratch[1024] ;
  int t = threadIdx.x ;
  scratch[t] = result[t] ;
  int offset = N ;
  while(offset > 2) {
    offset = offset/2 ;
    __syncthreads() ;
    if(t < offset)
      scratch[t] += scratch[t+offset] ;
  }
  __syncthreads() ;
  if(t==0)
    result[0] = scratch[0]+scratch[1] ;
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
  cudaError_t err = cudaMalloc(&d_pi,NBLOCKS*sizeof(double)) ;

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
  pisum<<<1,NBLOCKS>>>(NBLOCKS,d_pi) ;
  cudaEventRecord(stop) ;
  //  double cudapi[NTHREADS] ;
  double *cudapi = (double *)malloc(sizeof(double)*1) ;
  err = cudaMemcpy(&cudapi[0],d_pi,1*sizeof(double),cudaMemcpyDeviceToHost) ;
  if(err!= cudaSuccess) {
    cerr << "cudaMemcpy for d_pi failed" << endl ;
    cerr << "error = " << cudaGetErrorString(err) << endl;
  }
  cudaEventSynchronize(stop) ;
  float milliseconds = 0 ;
  cudaEventElapsedTime(&milliseconds,start,stop) ;
  cout<< "pi kernel execution time = " << milliseconds << " ms" << endl ;

  double cudapisum = 0 ;
  for(int i=0;i<1;++i)
    cudapisum += cudapi[i] ;
  computepi(N,&pi) ;
  cout << "pi = " << pi << " cudapi= " << cudapisum << endl ;
}

