#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_POINTS 100
#define BLOCK_SIZE 16

struct Point {
    float x;
    float y;
};

__global__ void computeDistances(Point *points, float *distances) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < NUM_POINTS && idy < NUM_POINTS) {
        int index = idy * NUM_POINTS + idx;
        float x1 = points[idx].x;
        float y1 = points[idx].y;
        float x2 = points[idy].x;
        float y2 = points[idy].y;
        distances[index] = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }
}

int main() {
    Point *h_points;
    float *h_distances;
    Point *d_points;
    float *d_distances;
    size_t points_size = NUM_POINTS * sizeof(Point);
    size_t distances_size = NUM_POINTS * NUM_POINTS * sizeof(float);

    // Allocate host memory for points and distances
    h_points = (Point*)malloc(points_size);
    h_distances = (float*)malloc(distances_size);

    // Generate 100 random points in a 100x100 cm grid
    for (int i = 0; i < NUM_POINTS; ++i) {
        h_points[i].x = (float)(rand() % 100);
        h_points[i].y = (float)(rand() % 100);
    }

    // Allocate device memory for points and distances
    cudaMalloc((void**)&d_points, points_size);
    cudaMalloc((void**)&d_distances, distances_size);

    // Copy points data from host to device
    cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE, (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the CUDA kernel
    computeDistances<<<grid_size, block_size>>>(d_points, d_distances);

    // Copy distances data from device to host
    cudaMemcpy(h_distances, d_distances, distances_size, cudaMemcpyDeviceToHost);

    // Print distances (for demonstration purposes)
    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < NUM_POINTS; ++j) {
            printf("%.2f ", h_distances[i * NUM_POINTS + j]);
        }
        printf("\n");
    }

    // Free allocated memory on both host and device
    free(h_points);
    free(h_distances);
    cudaFree(d_points);
    cudaFree(d_distances);

    return 0;
}

