import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

# CUDA kernel code for distance calculation
cuda_code = """
__global__ void compute_distances(Points *points, float *distances) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < 100 && idy < 100) {
        int index = idy * 100 + idx;
        float x1 = points[2 * idx];
        float y1 = points[2 * idx + 1];
        float x2 = points[2 * idy];
        float y2 = points[2 * idy + 1];
        distances[index] = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }
}
"""

# Compile CUDA code
mod = SourceModule(cuda_code)
compute_distances = mod.get_function("compute_distances")

# Generate 100 random points in the range [0, 100]
points = np.random.rand(100, 2) * 100

# Allocate memory on the GPU
points_gpu = cuda.mem_alloc(points.nbytes)
distances_gpu = cuda.mem_alloc(100 * 100 * 4)  # float32 = 4 bytes

# Copy data to GPU
cuda.memcpy_htod(points_gpu, points)

# Define block and grid sizes
block_size = (16, 16)
grid_size = ((100 + block_size[0] - 1) // block_size[0], (100 + block_size[1] - 1) // block_size[1])

# Launch the CUDA kernel
compute_distances(points_gpu, distances_gpu, block=block_size, grid=grid_size)

# Copy results back to CPU
distances = np.empty((100, 100), dtype=np.float32)
cuda.memcpy_dtoh(distances, distances_gpu)

# Visualize distances as a heatmap or matrix
plt.imshow(distances, cmap='viridis', origin='lower', extent=[0, 100, 0, 100])
plt.colorbar()
plt.title('Distances Between Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
