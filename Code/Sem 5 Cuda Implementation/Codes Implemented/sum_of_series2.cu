#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel to calculate the sum of a series of numbers
__global__ void sumOfSeries(float* result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    while (tid < n) {
        sum += tid;
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd(result, sum);
}

int main() {
    int n;

    // Prompt the user to enter the number of elements in the series
    std::cout << "Enter the number of elements in the series: ";
    std::cin >> n;

    int numThreadsPerBlock = 256;
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Allocate memory on the CPU to store the result
    float* h_result = new float;
    *h_result = 0.0f;

    // Allocate memory on the GPU to store the result
    float* d_result;
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the CUDA kernel
    sumOfSeries<<<numBlocks, numThreadsPerBlock>>>(d_result, n);

    // Ensure that the GPU has completed its work
    cudaDeviceSynchronize();

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time taken for execution: " << duration.count() << " nanoseconds" << std::endl;

    // Copy the result back to the CPU
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the sum of the series
    std::cout << "Sum of the series: " << *h_result << std::endl;

    // Cleanup
    delete h_result;
    cudaFree(d_result);

    return 0;
}

