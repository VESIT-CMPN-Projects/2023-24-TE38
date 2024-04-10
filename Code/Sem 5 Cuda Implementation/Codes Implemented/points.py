import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

# CUDA kernel to compute distances between points
cuda_code = """
__global__ void compute_distances(float *points, float *distances) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < 100 && idy < 100) {
        int index = idy * 100 + idx;
        float x1 = points[2 * idx];
        float y1 = points[2 * idx + 1];
        float x2 = points[2 * idy];
        float y2 = points[2 * idy + 1];
        distances[index] = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }
}
"""

mod = SourceModule(cuda_code)
compute_distances = mod.get_function("compute_distances")

# Generate 100 random points in the range [0, 100]
points = np.random.rand(100, 2) * 100

# Allocate memory on the GPU
points_gpu = cuda.mem_alloc(points.nbytes)
distances_gpu = cuda.mem_alloc(100 * 100 * 4)  # float32 = 4 bytes

# Copy data to the GPU
cuda.memcpy_htod(points_gpu, points)

# Define block and grid sizes
block_size = (16, 16)
grid_size = ((100 + block_size[0] - 1) // block_size[0], (100 + block_size[1] - 1) // block_size[1])

# Launch the CUDA kernel
compute_distances(points_gpu, distances_gpu, block=block_size, grid=grid_size)

# Copy the result back to the CPU
distances = np.empty((100, 100), dtype=np.float32)
cuda.memcpy_dtoh(distances, distances_gpu)

# Plotting the points on a 100x100 cm grid
plt.scatter(points[:, 0], points[:, 1])
plt.title('100 Points on a 100x100 cm Grid')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

# Print distances
print("Distances between points:")
print(distances)

