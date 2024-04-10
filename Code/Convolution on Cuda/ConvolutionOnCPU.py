import numpy as np 
from math import exp
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import time

def gaussian(sigma, kernel_size):
    m = kernel_size // 2
    n = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = exp(-((x - m) ** 2 + (y - n) ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)  # Normalize the kernel

def convolve(image, mask):
    image_rows, image_cols = image.shape
    mask_rows, mask_cols = mask.shape
    delta_rows = mask_rows // 2
    delta_cols = mask_cols // 2
    result = np.zeros_like(image)
    for i in range(image_rows):
        for j in range(image_cols):
            s = 0
            for k in range(mask_rows):
                for l in range(mask_cols):
                    i_k = i - k + delta_rows
                    j_l = j - l + delta_cols
                    if 0 <= i_k < image_rows and 0 <= j_l < image_cols:
                        s += mask[k, l] * image[i_k, j_l]
            result[i, j] = s
    return result

# Load and preprocess the image
image = np.asarray(ImageOps.grayscale(Image.open('image.jpg')))

# Define parameters
sigma = 4
kernel_size = 30

# Calculate Gaussian kernel
start_kernel = time.process_time()
kernel = gaussian(sigma, kernel_size)
end_kernel = time.process_time()

# Perform convolution
start_convolution = time.process_time()
result = convolve(image, kernel)
end_convolution = time.process_time()

# Print timing information
print("Time taken for Gaussian kernel calculation: {:.6f} seconds".format(end_kernel - start_kernel))
print("Time taken for convolution: {:.6f} seconds".format(end_convolution - start_convolution))

# Display images
plt.figure()
plt.imshow(image, cmap='gray' )
plt.title('Before Convolution without using CUDA')
plt.figure()
plt.imshow(result, cmap='gray')
plt.title('After Convolution without using CUDA')
plt.show()
