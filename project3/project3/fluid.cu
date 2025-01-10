#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <sys/time.h>

using namespace std;
//------------------------------------------------------------------------
// GPGPU Helper Routines

float *allocate_gpu(int nfloats, string name) {
    float *ret;
    cudaError_t err = cudaMalloc(&ret, nfloats * sizeof(float));
    if (err != cudaSuccess){
        cerr << "cudaMalloc for " << name << " failed" << endl;
        cerr << "error = " << cudaGetErrorString(err) << endl;
        exit(-1);
    }
    return ret;
}

void free_gpu(float *ptr){
    cudaError_t err = cudaFree(ptr);
}

void copy_gpu_to_cpu(float *gpu, float *cpu, int allocsize, string name){
    cudaError_t err = cudaMemcpy(cpu, gpu, allocsize * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        cerr << "cudaMemcpy for " << name << " failed" << endl;
        cerr << "error = " << cudaGetErrorString(err) << endl;
        exit(-1);
    }
}

void copy_cpu_to_gpu(float *cpu, float *gpu, int allocsize, string name){
    cudaError_t err = cudaMemcpy(gpu, cpu, allocsize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        cerr << "cudaMemcpy for " << name << " failed" << endl;
        cerr << "error = " << cudaGetErrorString(err) << endl;
        exit(-1);
    }
}

//------------------------------------------------------------------------
// Compute initial conditions for canonical Taylor-Green vortex problem
// This is an example of a CUDA kernel for the initialization routine
__global__ void setInitialConditions_kernel(float *p, float *u, float *v, float *w,
                                            int ni, int nj, int nk, int kstart,
                                            int iskip, int jskip, float L){
    const float l = 1.0;
    const float coef = 1.0;
    float dx = (1. / ni) * L;
    float dy = (1. / nj) * L;
    float dz = (1. / nk) * L;
    // The i iteration is assigned to thread blocks
    int i = blockIdx.x;
    // The k iteration is assigned to the threads within each thread block
    // (Note, this allows for coalesced memory accesses)
    int k = threadIdx.x;
    // The j iteration is performed within each thread block
    // But this could also have been added to the the thread dimensions
    for (int j = 0; j < nj; ++j){
        float x = 0.5 * dx + (i)*dx - 0.5 * L;
        float y = 0.5 * dy + j * dy - 0.5 * L;
        int offset = kstart + i * iskip + j * jskip;
        int indx = offset + k;
        float z = 0.5 * dz + k * dz - 0.5 * L;
        // 3-D taylor green vortex
        u[indx] = 1. * coef * sin(x / l) * cos(y / l) * cos(z / l);
        v[indx] = -1. * coef * cos(x / l) * sin(y / l) * cos(z / l);
        p[indx] = (1. / 16.) * coef * coef * (cos(2. * x / l) + cos(2. * y / l)) * (cos(2. * z / l) + 2.);
        w[indx] = 0;
    }
}


// Apply periodic boundary conditions at the boundary of the box
__global__ void copyPeriodic_kernel(float *p, float *u, float *v, float *w,
                int ni, int nj, int nk, int kstart, int iskip, int jskip){
    const int kskip = 1;
    // The i iteration is assigned to thread blocks
    int i = blockIdx.x;
    // The k iteration is assigned to the threads within each thread block
    int k = threadIdx.x;
    // copy the i periodic faces
    int indx = kstart + i * jskip + k * kskip;
    int offset = kstart + i * iskip;
    p[indx - iskip] = p[indx + (ni - 1) * iskip];
    p[indx - 2 * iskip] = p[indx + (ni - 2) * iskip];
    p[indx + ni * iskip] = p[indx];
    p[indx + (ni + 1) * iskip] = p[indx + iskip];

    u[indx - iskip] = u[indx + (ni - 1) * iskip];
    u[indx - 2 * iskip] = u[indx + (ni - 2) * iskip];
    u[indx + ni * iskip] = u[indx];
    u[indx + (ni + 1) * iskip] = u[indx + iskip];

    v[indx - iskip] = v[indx + (ni - 1) * iskip];
    v[indx - 2 * iskip] = v[indx + (ni - 2) * iskip];
    v[indx + ni * iskip] = v[indx];
    v[indx + (ni + 1) * iskip] = v[indx + iskip];

    w[indx - iskip] = w[indx + (ni - 1) * iskip];
    w[indx - 2 * iskip] = w[indx + (ni - 2) * iskip];
    w[indx + ni * iskip] = w[indx];
    w[indx + (ni + 1) * iskip] = w[indx + iskip];
    // copy the j periodic faces
    //for (int i = 0; i < ni; ++i){
        //int offset = kstart + i * iskip;
        //for (int k = 0; k < nk; ++k){
    indx = offset + k * kskip;
    p[indx - jskip] = p[indx + (nj - 1) * jskip];
    p[indx - 2 * jskip] = p[indx + (nj - 2) * jskip];
    p[indx + nj * jskip] = p[indx];
    p[indx + (nj + 1) * jskip] = p[indx + jskip];

    u[indx - jskip] = u[indx + (nj - 1) * jskip];
    u[indx - 2 * jskip] = u[indx + (nj - 2) * jskip];
    u[indx + nj * jskip] = u[indx];
    u[indx + (nj + 1) * jskip] = u[indx + jskip];

    v[indx - jskip] = v[indx + (nj - 1) * jskip];
    v[indx - 2 * jskip] = v[indx + (nj - 2) * jskip];
    v[indx + nj * jskip] = v[indx];
    v[indx + (nj + 1) * jskip] = v[indx + jskip];

    w[indx - jskip] = w[indx + (nj - 1) * jskip];
    w[indx - 2 * jskip] = w[indx + (nj - 2) * jskip];
    w[indx + nj * jskip] = w[indx];
    w[indx + (nj + 1) * jskip] = w[indx + jskip];

    // copy the k periodic faces
    indx = k * jskip + offset;
    p[indx - kskip] = p[indx + (nk - 1) * kskip];
    p[indx - 2 * kskip] = p[indx + (nk - 2) * kskip];
    p[indx + nk * kskip] = p[indx];
    p[indx + (nk + 1) * kskip] = p[indx + kskip];

    u[indx - kskip] = u[indx + (nk - 1) * kskip];
    u[indx - 2 * kskip] = u[indx + (nk - 2) * kskip];
    u[indx + nk * kskip] = u[indx];
    u[indx + (nk + 1) * kskip] = u[indx + kskip];

    v[indx - kskip] = v[indx + (nk - 1) * kskip];
    v[indx - 2 * kskip] = v[indx + (nk - 2) * kskip];
    v[indx + nk * kskip] = v[indx];
    v[indx + (nk + 1) * kskip] = v[indx + kskip];

    w[indx - kskip] = w[indx + (nk - 1) * kskip];
    w[indx - 2 * kskip] = w[indx + (nk - 2) * kskip];
    w[indx + nk * kskip] = w[indx];
    w[indx + (nk + 1) * kskip] = w[indx + kskip];
}


// Before summing up fluxes, zero out the residual term
__global__ void zeroResidual_kernel(float *presid, float *uresid, float *vresid, float *wresid,
                int ni, int nj, int nk, int kstart, int iskip, int jskip){
    // SERIAL PART for edge values:
    int i_edge[2] = {-1, ni};
    int k_edge[2] = {-1, nk};
    for (int i_idx = 0; i_idx < 2; i_idx++) {
        for (int k_idx = 0; k_idx < 2; k_idx++) {
            for (int j = -1; j < nj + 1; ++j){
                int offset = kstart + i_edge[i_idx]*iskip + j*jskip;
                int idx = k_edge[k_idx] + offset;
                presid[idx] = 0;
                uresid[idx] = 0;
                vresid[idx] = 0;
                wresid[idx] = 0;
            }
        }
    }
    // The i iteration is assigned to thread blocks
    int i = blockIdx.x;
    // The k iteration is assigned to the threads within each thread block
    int k = threadIdx.x;
    for (int j = -1; j < nj + 1; ++j){
        int offset = kstart + i*iskip + j*jskip;
        int idx = k + offset;
        presid[idx] = 0;
        uresid[idx] = 0;
        vresid[idx] = 0;
        wresid[idx] = 0;
    }
}

__global__ void computeResidual_x_kernel(float *presid, float *uresid, float *vresid, float *wresid,
                    const float *p, const float *u, const float *v, const float *w,
                    float eta, float nu, float dx, float dy, float dz,
                    int ni, int nj, int nk, int kstart, int iskip, int jskip) {
    const float vcoef = nu / dx;
    const float area = dy * dz;
    // i dimension goes in the +x coordinate direction
    int j = blockIdx.x;  // j iteration is assigned to thread blocks
    int k = threadIdx.x; // k iteration is assigned to the threads within each thread block
    // Loop through i faces of the mesh and compute fluxes in x direction
    // Add fluxes to cells that neighbor face
    for (int i = 0; i < ni+1; i++){
        int offset = kstart + i * iskip + j * jskip;
        const int indx = k + offset;
        // Compute the x direction inviscid flux; extract pressures from the stencil
        float ull = u[indx - 2 * iskip];
        float ul = u[indx - iskip];
        float ur = u[indx];
        float urr = u[indx + iskip];

        float vll = v[indx - 2 * iskip];
        float vl = v[indx - iskip];
        float vr = v[indx];
        float vrr = v[indx + iskip];

        float wll = w[indx - 2 * iskip];
        float wl = w[indx - iskip];
        float wr = w[indx];
        float wrr = w[indx + iskip];

        float pll = p[indx - 2 * iskip];
        float pl = p[indx - iskip];
        float pr = p[indx];
        float prr = p[indx + iskip];
        float pterm = (2. / 3.) * (pl + pr) - (1. / 12.) * (pl + pr + pll + prr);
        // x direction so the flux will be a function of u
        float udotn1 = ul + ur;
        float udotn2 = ul + urr;
        float udotn3 = ull + ur;
        float pflux = eta * ((2. / 3.) * udotn1 - (1. / 12.) * (udotn2 + udotn3));
        float uflux = ((1. / 3.) * (ul + ur) * udotn1 -
                        (1. / 24.) * ((ul + urr) * udotn2 + (ull + ur) * udotn3) + pterm);
        float vflux = ((1. / 3.) * (vl + vr) * udotn1 -
                        (1. / 24.) * ((vl + vrr) * udotn2 + (vll + vr) * udotn3));

        float wflux = ((1. / 3.) * (wl + wr) * udotn1 -
                        (1. / 24.) * ((wl + wrr) * udotn2 + (wll + wr) * udotn3));

        // Add in viscous fluxes integrate over face area
        pflux *= area;
        uflux = area * (uflux - vcoef * ((5. / 4.) * (ur - ul) - (1. / 12.) * (urr - ull)));
        vflux = area * (vflux - vcoef * ((5. / 4.) * (vr - vl) - (1. / 12.) * (vrr - vll)));
        wflux = area * (wflux - vcoef * ((5. / 4.) * (wr - wl) - (1. / 12.) * (wrr - wll)));

        presid[indx - iskip] -= pflux;
        uresid[indx - iskip] -= uflux;
        vresid[indx - iskip] -= vflux;
        wresid[indx - iskip] -= wflux;
        __syncthreads();
        presid[indx] += pflux;
        uresid[indx] += uflux;
        vresid[indx] += vflux;
        wresid[indx] += wflux;
    }
}

__global__ void computeResidual_y_kernel(float *presid, float *uresid, float *vresid, float *wresid,
                    const float *p, const float *u, const float *v, const float *w,
                    float eta, float nu, float dx, float dy, float dz,
                    int ni, int nj, int nk, int kstart, int iskip, int jskip) {
    const float vcoef = nu / dy;
    const float area = dx * dz;
    // j dimension goes in the +y coordinate direction
    int i = blockIdx.x;  // i iteration is assigned to thread blocks
    int k = threadIdx.x; // k iteration is assigned to the threads within each thread block
     // Loop through j faces of the mesh and compute fluxes in y direction
    // Add fluxes to cells that neighbor face
    for (int j = 0; j < nj+1; j++){
        int offset = kstart + i * iskip + j * jskip;
        const int indx = k + offset;
        // Compute the y direction inviscid flux
        // extract pressures and velocity from the stencil
        float ull = u[indx - 2 * jskip];
        float ul = u[indx - jskip];
        float ur = u[indx];
        float urr = u[indx + jskip];

        float vll = v[indx - 2 * jskip];
        float vl = v[indx - jskip];
        float vr = v[indx];
        float vrr = v[indx + jskip];

        float wll = w[indx - 2 * jskip];
        float wl = w[indx - jskip];
        float wr = w[indx];
        float wrr = w[indx + jskip];

        float pll = p[indx - 2 * jskip];
        float pl = p[indx - jskip];
        float pr = p[indx];
        float prr = p[indx + jskip];
        float pterm = (2. / 3.) * (pl + pr) - (1. / 12.) * (pl + pr + pll + prr);
        // y direction so the flux will be a function of v
        float udotn1 = vl + vr;
        float udotn2 = vl + vrr;
        float udotn3 = vll + vr;
        float pflux = eta * ((2. / 3.) * udotn1 - (1. / 12.) * (udotn2 + udotn3));
        float uflux = ((1. / 3.) * (ul + ur) * udotn1 -
                        (1. / 24.) * ((ul + urr) * udotn2 + (ull + ur) * udotn3));

        float vflux = ((1. / 3.) * (vl + vr) * udotn1 -
                        (1. / 24.) * ((vl + vrr) * udotn2 + (vll + vr) * udotn3) + pterm);

        float wflux = ((1. / 3.) * (wl + wr) * udotn1 -
                        (1. / 24.) * ((wl + wrr) * udotn2 + (wll + wr) * udotn3));

        // Add in viscous fluxes integrate over face area
        pflux *= area;
        uflux = area * (uflux - vcoef * ((5. / 4.) * (ur - ul) - (1. / 12.) * (urr - ull)));
        vflux = area * (vflux - vcoef * ((5. / 4.) * (vr - vl) - (1. / 12.) * (vrr - vll)));
        wflux = area * (wflux - vcoef * ((5. / 4.) * (wr - wl) - (1. / 12.) * (wrr - wll)));
        presid[indx - jskip] -= pflux;
        uresid[indx - jskip] -= uflux;
        vresid[indx - jskip] -= vflux;
        wresid[indx - jskip] -= wflux;
        __syncthreads();
        presid[indx] += pflux;
        uresid[indx] += uflux;
        vresid[indx] += vflux;
        wresid[indx] += wflux;
    }
}

__global__ void computeResidual_z_kernel(float *presid, float *uresid, float *vresid, float *wresid,
                    const float *p, const float *u, const float *v, const float *w,
                    float eta, float nu, float dx, float dy, float dz,
                    int ni, int nj, int nk, int kstart, int iskip, int jskip) {
    const float vcoef = nu / dz;
    const float area = dx * dy;
    // k dimension goes in the +z coordinate direction
    const int kskip = 1;
    int j = blockIdx.x;  // j iteration is assigned to thread blocks
    int k = threadIdx.x; // k iteration is assigned to the threads within each thread block
    // Loop through k faces of the mesh and compute fluxes in z direction
    // Add fluxes to cells that neighbor face
    for (int i = 0; i < ni; i++){
        int offset = kstart + i*iskip + j*jskip;
        const int indx = k + offset;
        // Compute the y direction inviscid flux
        // extract pressures and velocity from the stencil
        float ull = u[indx - 2 * kskip];
        float ul = u[indx - kskip];
        float ur = u[indx];
        float urr = u[indx + kskip];

        float vll = v[indx - 2 * kskip];
        float vl = v[indx - kskip];
        float vr = v[indx];
        float vrr = v[indx + kskip];

        float wll = w[indx - 2 * kskip];
        float wl = w[indx - kskip];
        float wr = w[indx];
        float wrr = w[indx + kskip];

        float pll = p[indx - 2 * kskip];
        float pl = p[indx - kskip];
        float pr = p[indx];
        float prr = p[indx + kskip];
        float pterm = (2. / 3.) * (pl + pr) - (1. / 12.) * (pl + pr + pll + prr);
        // y direction so the flux will be a function of v
        float udotn1 = wl + wr;
        float udotn2 = wl + wrr;
        float udotn3 = wll + wr;
        float pflux = eta*((2./3.)*udotn1 - (1./12.)*(udotn2 + udotn3));
        float uflux = ((1./3.)*(ul + ur)*udotn1 - (1./24.)*((ul + urr)*udotn2 + (ull + ur)*udotn3));
        float vflux = ((1./3.)*(vl + vr)*udotn1 - (1./24.)*((vl + vrr)*udotn2 + (vll + vr)*udotn3));
        float wflux = ((1./3.)*(wl + wr)*udotn1 - (1./24.)*((wl + wrr)*udotn2 + (wll + wr)*udotn3) + pterm);

        // Add in viscous fluxes integrate over face area
        pflux *= area;
        uflux = area * (uflux - vcoef * ((5. / 4.) * (ur - ul) - (1. / 12.) * (urr - ull)));
        vflux = area * (vflux - vcoef * ((5. / 4.) * (vr - vl) - (1. / 12.) * (vrr - vll)));
        wflux = area * (wflux - vcoef * ((5. / 4.) * (wr - wl) - (1. / 12.) * (wrr - wll)));
        presid[indx - kskip] -= pflux;
        uresid[indx - kskip] -= uflux;
        vresid[indx - kskip] -= vflux;
        wresid[indx - kskip] -= wflux;
        __syncthreads();
        presid[indx] += pflux;
        uresid[indx] += uflux;
        vresid[indx] += vflux;
        wresid[indx] += wflux;
    }
}

// Compute the residue which is represent the computed rate of change for the
// pressure and the three components of the velocity vector denoted (u,v,w)
void computeResidual(float *presid, float *uresid, float *vresid, float *wresid,
                    const float *p, const float *u, const float *v, const float *w,
                    float eta, float nu, float dx, float dy, float dz,
                    int ni, int nj, int nk, int kstart, int iskip, int jskip) {
    // i dimension goes in the +x coordinate direction
    computeResidual_x_kernel<<<nj, nk>>>(presid, uresid, vresid, wresid,
                        p, u, v, w, eta, nu, dx, dy, dz, ni, nj, nk, kstart, iskip, jskip);
    computeResidual_y_kernel<<<ni, nk>>>(presid, uresid, vresid, wresid,
                        p, u, v, w, eta, nu, dx, dy, dz, ni, nj, nk, kstart, iskip, jskip);
    computeResidual_z_kernel<<<nj, nk+1>>>(presid, uresid, vresid, wresid,
                        p, u, v, w, eta, nu, dx, dy, dz, ni, nj, nk, kstart, iskip, jskip);
}


// Calculate the stable timestep considering inviscid and viscous terms
__global__ void computeStableTimestep_kernel(float *scratch, const float *u, const float *v, const float *w,
                            float cfl, float eta, float nu, float dx, float dy, float dz,
                            int ni, int nj, int nk, int kstart, int iskip, int jskip){
    float minDt = 1e30;
    int i = blockIdx.x;
    int k = threadIdx.x;
    for (int j = 0; j < nj; ++j){
        int offset = kstart + i*iskip + j*jskip;
        const int indx = k + offset;
        // inviscid timestep
        const float maxu2 = max(u[indx] * u[indx], max(v[indx] * v[indx], w[indx] * w[indx]));
        const float af = sqrt(maxu2 + eta);
        const float maxev = sqrt(maxu2) + af;
        const float sum = maxev * (1. / dx + 1. / dy + 1. / dz);
        minDt = min(minDt, cfl / sum);
        // viscous stable timestep
        const float dist = min(dx, min(dy, dz));
        minDt = min(minDt, 0.2 * cfl * dist * dist / nu);
    }
    // We store the sums over the k iteration into the scratch array
    scratch[i * nk + k] = minDt;
}

__global__ void minKernel(float *mins){
    // each thread within the thread block will used shared memory to
    // compute the min within the thread block using __syncthreads().
    // The final minimization over thread blocks will be performed by the CPU
    int t = threadIdx.x;
    int b = blockIdx.x;
    __shared__ float scratch[1024];
    // There will be 1024 threads used to get the most parallel operations possible.
    // First we copy the value we are combining into shared memory so that threads can work
    // on minimizing the results without touching main memory
    scratch[t] = mins[b * blockDim.x + t]; // load shared memory
    // implement a tree based reduction strategy where the size of scratch is halved each step.
    int nthreads = blockDim.x;
    int offset = nthreads;
    // offset is set to the size of the scratch array
    while (offset > 2) { // sum into shared
        offset >>= 1; // offset = offset / 2
        __syncthreads();
        // check if last update to scratch has been completed by all threads in block
        if (t < offset) // if our thread is writing to memory, write minimum
            scratch[t] = min(scratch[t], scratch[t + offset]);
    }
    __syncthreads();
    // Final step writes result from each thread block into mins array
    if (t == 0)
        mins[b] = min(scratch[0], scratch[1]);
}


// Compute the fluid kinetic energy contained within the simulation domain
// This is part of a 2 part kernel for summing the kinetic enery for the mesh
__global__ void integrateKineticEnergy_kernel(float *scratch,
                                            const float *u, const float *v, const float *w,
                                            float dx, float dy, float dz, int ni, int nj, int nk,
                                            int kstart, int iskip, int jskip) {
    double vol = dx * dy * dz;
    double sum = 0;
    // Threads are allocated to iterations as was done in initial conditions
    int i = blockIdx.x;
    int k = threadIdx.x;
    for (int j = 0; j < nj; ++j){
        int offset = kstart + i * iskip + j * jskip;
        const int indx = k + offset;
        const float udotu = u[indx] * u[indx] + v[indx] * v[indx] + w[indx] * w[indx];
        sum += 0.5 * vol * udotu;
    }
    // We store the sums over the k iteration into the scratch array
    scratch[i * nk + k] = sum;
}

__global__ void sumKernel(float *sum){
    // each thread in a thread block will used shared memory to compute sums
    // within the thread block (using a __syncthreads() call).
    // final summation over thread blocks will be performed by the CPU
    int t = threadIdx.x;
    int b = blockIdx.x;
    __shared__ float scratch[1024];
    scratch[t] = sum[b * blockDim.x + t]; // load shared memory
    // implements a tree based summing strategy where the size of scratch is halved each step
    int nthreads = blockDim.x;
    int offset = nthreads;
    // offset is set to the size of the scratch array
    while (offset > 2) {
        // compute offset to paired number
        offset >>= 1; // right bitshift by 1
        __syncthreads();
        // checks if previous update to scratch has been completed by all threads in block
        if (t < offset) // if our thread is writing to memory ,write sum
            scratch[t] += scratch[t + offset];
    }
    __syncthreads();
    // Final step writes result from each thread block into sum array
    if (t == 0)
        sum[b] = scratch[0] + scratch[1];
}

// Perform a weighted sum of three arrays
// Note, the last weight is used for the input array (no aliasing)
__global__ void weightedSum3_kernel(float *uout, float w1, const float *u1, float w2, const float *u2, float w3,
                int ni, int nj, int nk, int kstart, int iskip, int jskip) {
    int i = blockIdx.x;
    int k = threadIdx.x;
    for(int j = 0; j < nj; ++j) {
        int offset = kstart + i*iskip + j*jskip;
        const int idx = k + offset;
        uout[idx] = w1*u1[idx] + w2*u2[idx] + w3*uout[idx];
    }
}

int parse_arg(char* av) {
    const char* flag_str[11] = { "-n","-ni","-nj","-nk","-L","-nu","-refVel","-stopTime","-cflmax","-outfile","-o" };
    for (int i = 0; i < 11; i++) {
        if (!strcmp(av, flag_str[i]))
            return i;
    }
    return -1;
}

int main(int ac, char *av[]){
    // Default Simulation Parameters
    // Dimensions of the simulation mesh
    int ni = 32;
    int nj = 32;
    int nk = 32;
    // Length of the cube
    float L = 6.28318530718;
    // fluid viscosity
    float nu = 0.000625;
    // Reference velocity used for artificial compressibility apprach
    float refVel = 10;
    // Simulation stopping time
    float stopTime = 20;
    // Coefficient used to compute stable timestep
    float cflmax = 1.9;

    string outfile = "fke.dat";
    // parse command line arguments
    while (ac >= 2 && av[1][0] == '-') {
        int flag = parse_arg(av[1]);
        if ((ac >= 3) && (flag >= 0) && (flag < 11)) {
            switch (flag) {
            case 0:
                ni = atoi(av[2]);
                nj = ni;
                nk = ni;
                break;
            case 1:
                ni = atoi(av[2]); break;
            case 2:
                nj = atoi(av[2]); break;
            case 3:
                nk = atoi(av[2]); break;
            case 4:
                L = atof(av[2]); break;
            case 5:
                nu = atof(av[2]); break;
            case 6:
                refVel = atof(av[2]); break;
            case 7:
                stopTime = atof(av[2]); break;
            case 8:
                cflmax = atof(av[2]); break;
            case 9: // -outfile and -o have the same outcome
            case 10:
                outfile = av[2]; break;
            }
            av += 2;
            ac -= 2;
        }
        else {
            cerr << "unknown command line argument '" << av[1] << "'" << endl;
            av += 1;
            ac -= 1;
            exit(-1);
        }
    }
    // File to save the fluid kinetic energy history
    ofstream ke_file(outfile.c_str(), ios::trunc);

    // Eta is a artificial compressibility parameter to the numerical scheme
    float eta = refVel * refVel;

    // The mesh cell sizes
    float dx = L / ni;
    float dy = L / nj;
    float dz = L / nk;

    struct timeval tval_start, tval_end, tval_elapsed;
    gettimeofday(&tval_start, 0);
    //  Allocate a 3-D mesh with enough space ghost cells two layers thick on each
    //  side of the mesh.
    int allocsize = (ni + 4) * (nj + 4) * (nk + 4);
    cout.precision(5);
    cout << "allocating " << ((allocsize * 4 * 3 * sizeof(float)) >> 10) << " k bytes for fluid computation" << endl;
    // Fluid pressure and velocity

    /* vector<float> p(allocsize);
    vector<float> u(allocsize);
    vector<float> v(allocsize);
    vector<float> w(allocsize);
    // scratch space used to estimate the next timestep values in
    // time integration
    vector<float> pnext(allocsize);
    vector<float> unext(allocsize);
    vector<float> vnext(allocsize);
    vector<float> wnext(allocsize);
    // scratch space to store residual
    vector<float> presid(allocsize);
    vector<float> uresid(allocsize);
    vector<float> vresid(allocsize);
    vector<float> wresid(allocsize);*/
    // Allocate on cuda side
    float *p_cuda = allocate_gpu(allocsize, "p");
    float *u_cuda = allocate_gpu(allocsize, "u");
    float *v_cuda = allocate_gpu(allocsize, "v");
    float *w_cuda = allocate_gpu(allocsize, "w");
    // scratch space used to estimate the next timestep values in
    // time integration
    float *pnext_cuda = allocate_gpu(allocsize, "pnext");
    float *unext_cuda = allocate_gpu(allocsize, "unext");
    float *vnext_cuda = allocate_gpu(allocsize, "vnext");
    float *wnext_cuda = allocate_gpu(allocsize, "wnext");
    // scratch space to store residual
    float *presid_cuda = allocate_gpu(allocsize, "presid");
    float *uresid_cuda = allocate_gpu(allocsize, "uresid");
    float *vresid_cuda = allocate_gpu(allocsize, "vresid");
    float *wresid_cuda = allocate_gpu(allocsize, "wresid");

    float *scratch_cuda = allocate_gpu(ni * nk, "scratch");

    int iskip = (nk + 4) * (nj + 4);
    int jskip = (nk + 4);
    int kstart = 2 * iskip + 2 * jskip + 2;

    // Setup initial conditions
    // no need to copy cuda vectors to the gpu since they're empty here and aren't used in computation
    setInitialConditions_kernel<<<ni, nk>>>(p_cuda, u_cuda, v_cuda, w_cuda,
                                            ni, nj, nk, kstart, iskip, jskip, L);

    // Find initial integrated fluid kinetic energy to monitor solution
    integrateKineticEnergy_kernel<<<ni, nk>>>(scratch_cuda, u_cuda, v_cuda, w_cuda, dx, dy, dz,
                                            ni, nj, nk, kstart, iskip, jskip);
    int ntot = ni*nk;
    int nblocks = ntot >> 10; // ntot/(2^10)
    sumKernel<<<nblocks, 1024>>>(scratch_cuda);
    vector<float> tmp(nblocks);
    copy_gpu_to_cpu(scratch_cuda, &tmp[0], nblocks, "sum");
    float kprev = tmp[0];
    for (int i = 1; i < nblocks; ++i)
        kprev += tmp[i];

    // We use this scaling parameter so we can plot normalized kinetic energy
    float kscale = 1. / kprev;

    // Starting simulation time
    float simTime = 0;
    int iter = 0;

    //float dt = computeStableTimestep(&u[0], &v[0], &w[0], cflmax, eta, nu, dx, dy, dz,
                                    //ni, nj, nk, kstart, iskip, jskip);
    computeStableTimestep_kernel<<<ni, nk>>>(scratch_cuda, u_cuda, v_cuda, w_cuda, cflmax, eta, nu, dx, dy, dz,
                                    ni, nj, nk, kstart, iskip, jskip);
    minKernel<<<nblocks, 1024>>>(scratch_cuda);
    copy_gpu_to_cpu(scratch_cuda, &tmp[0], nblocks, "min");
    float dt = tmp[0];
    for (int i = 1; i < nblocks; ++i)
        dt = min(tmp[i], dt);

    // begin Runge-Kutta 3rd Order Time Integration
    while (simTime < stopTime){
        // copy data to the ghost cells to implement periodic boundary conditions
        copyPeriodic_kernel<<<ni, nk>>>(p_cuda, u_cuda, v_cuda, w_cuda, ni, nj, nk, kstart, iskip, jskip);

        // Zero out the residual function
        // no need to move residuals to gpu first since they're not used in computation
        zeroResidual_kernel<<<ni, nk>>>(presid_cuda, uresid_cuda, vresid_cuda, wresid_cuda,
                                        ni, nj, nk, kstart, iskip, jskip);

        // Compute the residual, these will be used to compute the rates of change
        // of pressure and velocity components
        computeResidual(&presid_cuda[0], &uresid_cuda[0], &vresid_cuda[0], &wresid_cuda[0],
                        &p_cuda[0], &u_cuda[0], &v_cuda[0], &w_cuda[0], eta, nu, dx, dy, dz,
                        ni, nj, nk, kstart, iskip, jskip);

        // First Step of the Runge-Kutta time integration
        // unext = u^n + dt/vol*L(u^n)
        weightedSum3_kernel<<<ni,nk>>>(pnext_cuda, 1.0, p_cuda, dt/(dx*dy*dz), presid_cuda, 0.0,
                                ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(unext_cuda, 1.0, u_cuda, dt/(dx*dy*dz), uresid_cuda, 0.0,
                                ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(vnext_cuda, 1.0, v_cuda, dt/(dx*dy*dz), vresid_cuda, 0.0,
                                ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(wnext_cuda, 1.0, w_cuda, dt/(dx*dy*dz), wresid_cuda, 0.0,
                                ni, nj, nk, kstart, iskip, jskip);

        // Now we are evaluating a residual a second time as part of the
        // third order time integration.  The residual is evaluated using
        // the first estimate of the time integrated solution that is found
        // in next version of the variables computed by the previous
        // wieghtedSum3 calls.

        // Now we are on the second step of the Runge-Kutta time integration
        copyPeriodic_kernel<<<ni, nk>>>(pnext_cuda, unext_cuda, vnext_cuda, wnext_cuda,
                                        ni, nj, nk, kstart, iskip, jskip);

        zeroResidual_kernel<<<ni, nk>>>(presid_cuda, uresid_cuda, vresid_cuda, wresid_cuda,
                    ni, nj, nk, kstart, iskip, jskip);

        computeResidual(&presid_cuda[0], &uresid_cuda[0], &vresid_cuda[0], &wresid_cuda[0],
                        &pnext_cuda[0], &unext_cuda[0], &vnext_cuda[0], &wnext_cuda[0],
                        eta, nu, dx, dy, dz, ni, nj, nk, kstart, iskip, jskip);

        // Second Step of the Runge-Kutta time integration
        // unext = 3/4 u^n + 1/4 u_next + (1/4)*(dt/vol)*L(unext)
        weightedSum3_kernel<<<ni,nk>>>(pnext_cuda, 3./4., p_cuda, dt/(4.*dx*dy*dz), presid_cuda,
                                        1./4., ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(unext_cuda, 3./4., u_cuda, dt/(4.*dx*dy*dz), uresid_cuda,
                                        1./4., ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(vnext_cuda, 3./4., v_cuda, dt/(4.*dx*dy*dz), vresid_cuda,
                                        1./4., ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(wnext_cuda, 3./4., w_cuda, dt/(4.*dx*dy*dz), wresid_cuda,
                                        1./4., ni, nj, nk, kstart, iskip, jskip);

        // Now we are evaluating the final step of the Runge-Kutta time integration
        // so we need to revaluate the residual on the pnext values

        copyPeriodic_kernel<<<ni, nk>>>(pnext_cuda, unext_cuda, vnext_cuda, wnext_cuda,
                                        ni, nj, nk, kstart, iskip, jskip);


        zeroResidual_kernel<<<ni, nk>>>(presid_cuda, uresid_cuda, vresid_cuda, wresid_cuda,
                    ni, nj, nk, kstart, iskip, jskip);

        computeResidual(&presid_cuda[0], &uresid_cuda[0], &vresid_cuda[0], &wresid_cuda[0],
                        &pnext_cuda[0], &unext_cuda[0], &vnext_cuda[0], &wnext_cuda[0],
                        eta, nu, dx, dy, dz, ni, nj, nk, kstart, iskip, jskip);

        // Third Step of the Runge-Kutta time integration
        // u^{n+1} = 1/3 u^n + 2/3 unext + (2/3)*(dt/vol)*L(unext)
        // Note, here we are writing the result into the previous timestep so that we will
        //be ready to proceed to the next iteration when this step is finished.
        weightedSum3_kernel<<<ni,nk>>>(p_cuda, 2./3., pnext_cuda, 2.*dt/(3.*dx*dy*dz), presid_cuda,
                                        1./3., ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(u_cuda, 2./3., unext_cuda, 2.*dt/(3.*dx*dy*dz), uresid_cuda,
                                        1./3., ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(v_cuda, 2./3., vnext_cuda, 2.*dt/(3.*dx*dy*dz), vresid_cuda,
                                        1./3., ni, nj, nk, kstart, iskip, jskip);
        weightedSum3_kernel<<<ni,nk>>>(w_cuda, 2./3., wnext_cuda, 2.*dt/(3.*dx*dy*dz), wresid_cuda,
                                        1./3., ni, nj, nk, kstart, iskip, jskip);

        // Update the simulation time
        simTime += dt;
        iter++;

        // Collect information on the state of kinetic energy in the system
        integrateKineticEnergy_kernel<<<ni, nk>>>(scratch_cuda, u_cuda, v_cuda, w_cuda, dx, dy, dz,
                                            ni, nj, nk, kstart, iskip, jskip);
        sumKernel<<<nblocks, 1024>>>(scratch_cuda);
        vector<float> tmp(nblocks);
        copy_gpu_to_cpu(scratch_cuda, &tmp[0], nblocks, "sum");
        float knext = tmp[0];
        for (int i = 1; i < nblocks; ++i)
            knext += tmp[i];

        // write out the data for post processing analysis
        ke_file << simTime << " " << kscale * knext << " " << -kscale * (knext - kprev) / dt << endl;
        // Every 128 iterations report the state so we can observe progress of
        // the simulation
        if ((iter & 0x7f) == 0)
            cout << "ke: " << simTime << ' ' << kscale * knext << endl;
        // keep track of the change in kinetic energy over timesteps so we can plot
        // the derivative of the kinetic energy with time.
        kprev = knext;
    }

    gettimeofday(&tval_end, 0);
    timersub(&tval_end, &tval_start, &tval_elapsed);
    double milliseconds = tval_elapsed.tv_sec * 1000.0 + tval_elapsed.tv_usec * 0.001;
    cout << "fluid execution time = " << milliseconds << " ms" << endl;

    cout << "time per cell per timestep = " << 1e6 * milliseconds / (double(ni) * double(nj) * double(nk) * double(iter)) << " ns" << endl;
    cout << "finished with iter = " << iter << endl;
    // release gpu memory
    free_gpu(p_cuda);
    free_gpu(u_cuda);
    free_gpu(v_cuda);
    free_gpu(w_cuda);
    // scratch space used to estimate the next timestep values in
    // time integration
    free_gpu(pnext_cuda);
    free_gpu(unext_cuda);
    free_gpu(vnext_cuda);
    free_gpu(wnext_cuda);
    // scratch space to store residual
    free_gpu(presid_cuda);
    free_gpu(uresid_cuda);
    free_gpu(vresid_cuda);
    free_gpu(wresid_cuda);

    free_gpu(scratch_cuda);
    return 0;
}