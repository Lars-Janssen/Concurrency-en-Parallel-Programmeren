/*
    Lars Janssen 12882712, Aron de Ruijter

    Implementation of a wave equation simulation, parallelized on the GPU using
    CUDA.
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


/*
    This is the kernel, which runs on the GPU. It checks if the point we want
    to change is not the first or last one, as they stay 0. Then we update it
    and change the old and current array accordingly.
*/
__global__ void WaveKernel(double* old_GPU, double* current_GPU,
                                double* next_GPU, int i_max, const long t_max)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    for (long t = 0; t < t_max; t++)
    {
        if(index > 0 && index < i_max - 1)
        {
            next_GPU[index] = 2 * current_GPU[index] - old_GPU[index] + 0.15 *
            (current_GPU[index-1] - (2 * current_GPU[index] - current_GPU[index+1]));
            old_GPU[index] = current_GPU[index];
            current_GPU[index] = next_GPU[index];
        }
        __syncthreads();
    }
}


/*
    Function that prepares and copies data to the GPU, runs the kernel, and then
    copies the result back.
*/
 void vectorAddCuda(int n, double* old_array, double* current_array, double* next_array, int block_size, const long t_max) {

    /*
        This allocates the vectors on the GPU, each time checking if we were
        successfull.
    */
    double* old_GPU = NULL;
    checkCudaCall(cudaMalloc((void **) &old_GPU, n * sizeof(double)));
    if (old_GPU == NULL) {
        cerr << "Could not allocate the old array on GPU." << endl;
        return;
    }
    double* current_GPU = NULL;
    checkCudaCall(cudaMalloc((void **) &current_GPU, n * sizeof(double)));
    if (current_GPU == NULL) {
        checkCudaCall(cudaFree(old_GPU));
        cerr << "Could not allocate the current array on GPU." << endl;
        return;
    }
    double* next_GPU = NULL;
    checkCudaCall(cudaMalloc((void **) &next_GPU, n * sizeof(double)));
    if (next_GPU == NULL) {
        checkCudaCall(cudaFree(old_GPU));
        checkCudaCall(cudaFree(current_GPU));
        cerr << "Could not allocate the next array on GPU." << endl;
        return;
    }

    /*
        This copies the old array and the current array to the GPU.
    */
    checkCudaCall(cudaMemcpy(old_GPU, old_array, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(current_GPU, current_array, n*sizeof(double), cudaMemcpyHostToDevice));

    /*
        This executes the wave kernel for every timestep.
    */
    WaveKernel<<<n/block_size + 1, block_size>>>(old_GPU, current_GPU, next_GPU, n, t_max);

    /*
        This checks whether the kernel invocation was succesful.
    */
    checkCudaCall(cudaGetLastError());

    /*
        This copies the result back to the host
    */
    checkCudaCall(cudaMemcpy(current_array, current_GPU, n * sizeof(double), cudaMemcpyDeviceToHost));

    /*
        This cleans up the data on the GPU.
    */
    checkCudaCall(cudaFree(old_GPU));
    checkCudaCall(cudaFree(current_GPU));
    checkCudaCall(cudaFree(next_GPU));

}


/*
    Function that will simulate the wave equation, parallelized using CUDA.
    i_max: how many data points are on a single wave
    t_max: how many iterations the simulation should run
    num_threads: how many threads to use (excluding the main threads)
    old_array: array of size i_max filled with data for t-1
    current_array: array of size i_max filled with data for t
    next_array: array of size i_max. You should fill this with t+1
*/
double *simulate(const long i_max, const long t_max, const long block_size,
                 double *old_array, double *current_array, double *next_array) {
    vectorAddCuda(i_max, old_array, current_array, next_array, block_size, t_max);
    return current_array;
}
