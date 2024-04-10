#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <iostream>

#define N 100000 // Size of the square matrices

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(long int* A, long int* B, long int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        long int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    long int* h_A, * h_B, * h_C; // Host matrices
    long int* d_A, * d_B, * d_C; // Device matrices

    // Allocate host memory
    h_A = (long int*)malloc(N * N * sizeof(long int));
    h_B = (long int*)malloc(N * N * sizeof(long int));
    h_C = (long int*)malloc(N * N * sizeof(long int));

    // Initialize matrices A and B with random data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(long int));
    cudaMalloc((void**)&d_B, N * N * sizeof(long int));
    cudaMalloc((void**)&d_C, N * N * sizeof(long int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(long int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the matrix multiplication kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result back from the device to the host
    cudaMemcpy(h_C, d_C, N * N * sizeof(long int), cudaMemcpyDeviceToHost);

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    // Print the result (matrix C)
    // printf("Matrix C (Result):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // printf("%ld\t", h_C[i * N + j]);
        }
        // printf("\n");
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

