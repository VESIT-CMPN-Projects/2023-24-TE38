import time
import random

def generate_random_matrix(rows, cols):
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

def matrix_multiplication(matrix1, matrix2):
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

# Get user input for the size of the matrices
n = int(input("Enter the size of the matrices (n): "))

# Generate random matrices of size n x n
matrix_A = generate_random_matrix(n, n)
matrix_B = generate_random_matrix(n, n)

# Perform matrix multiplication and measure time
start_time = time.time()
result_matrix = matrix_multiplication(matrix_A, matrix_B)
end_time = time.time()



elapsed_time_ns = (end_time - start_time) * 1e9  # Convert seconds to nanoseconds
print(f"Time taken for matrix multiplication: {elapsed_time_ns:.2f} nanoseconds")

