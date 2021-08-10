/*
 * simulate.cu
 *
 * Implementation of a wave equation simulation, parallelized on the GPU using
 * CUDA.
 *
 * You are supposed to edit this file with your implementation, and this file
 * only.
 *
 */

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "simulate.hh"

using namespace std;


/* Utility function, use to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 *
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
**/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}


/* The kernel, which runs on the GPU. */
__global__ void vectorAddKernel(double* old_GPU, double* current_GPU, double* next_GPU, int i_max) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index != 0 && index != i_max)
    {
        next_GPU[index] = 2 * current_GPU[index] - old_GPU[index] + 0.15 *
        (current_GPU[index-1] - (2 * current_GPU[index] - current_GPU[index+1]));
    }
}

/* Function that prepares & copies data to the GPU, runs the kernel, and then
 * copies the result. back. */
 void vectorAddCuda(int n, double* old_array, double* current_array, double* next_array, int block_size, const long t_max) {
    int threadBlockSize = block_size;

    // Allocate the vectors on the GPU, each time checking if we were successfull
    double* old_GPU = NULL;
    checkCudaCall(cudaMalloc((void **) &old_GPU, n * sizeof(double)));
    if (old_GPU == NULL) {
        cerr << "Could not allocate old array on GPU." << endl;
        return;
    }
    double* current_GPU = NULL;
    checkCudaCall(cudaMalloc((void **) &current_GPU, n * sizeof(double)));
    if (current_GPU == NULL) {
        checkCudaCall(cudaFree(old_GPU));
        cerr << "Could not allocate current array on GPU." << endl;
        return;
    }
    double* next_GPU = NULL;
    checkCudaCall(cudaMalloc((void **) &next_GPU, n * sizeof(double)));
    if (next_GPU == NULL) {
        checkCudaCall(cudaFree(old_GPU));
        checkCudaCall(cudaFree(current_GPU));
        cerr << "Could not allocate next array on GPU." << endl;
        return;
    }

    // Copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(old_GPU, old_array, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(current_GPU, current_array, n*sizeof(double), cudaMemcpyHostToDevice));

    // Execute the vector-add kernel
    for (int t = 0; t < t_max; t++)
    {
        vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(old_GPU, current_GPU, next_GPU, n);
        checkCudaCall(cudaMemcpy(old_GPU, current_GPU, n*sizeof(double), cudaMemcpyHostToDevice));
        checkCudaCall(cudaMemcpy(current_GPU, next_array, n*sizeof(double), cudaMemcpyHostToDevice));
    }

    // Check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // Copy result back to host
    checkCudaCall(cudaMemcpy(next_array, next_GPU, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup GPU-side data
    checkCudaCall(cudaFree(old_GPU));
    checkCudaCall(cudaFree(current_GPU));
    checkCudaCall(cudaFree(next_GPU));

}

/* Function that will simulate the wave equation, parallelized using CUDA.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 *
 */
double *simulate(const long i_max, const long t_max, const long block_size,
                 double *old_array, double *current_array, double *next_array) {

    vectorAddCuda(i_max, old_array, current_array, next_array, block_size, t_max);

    return current_array;
}
