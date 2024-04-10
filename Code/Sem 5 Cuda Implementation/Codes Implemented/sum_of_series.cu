#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to calculate the sum of a series of numbers
__global__ void sumOfSeries(float* result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    while (tid < n) {
        sum += 1;
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd(result, sum);
}

int main() {
    int n = 1000000;  // Number of elements in the series
    int numThreadsPerBlock = 256; //confirm 
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Allocate memory on the CPU to store the result
    float* h_result = new float;
    *h_result = 0.0f;

    // Allocate memory on the GPU to store the result
    float* d_result;
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    sumOfSeries<<<numBlocks, numThreadsPerBlock>>>(d_result, n);

    // Copy the result back to the CPU
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the sum of the series
    std::cout << "Sum of the series: " << *h_result << std::endl;

    // Cleanup
    delete h_result;
    cudaFree(d_result);

    return 0;
}

