import random
import math
import matplotlib.pyplot as plt

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Generate 10 random points within a 100x100 box
points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]

# Calculate distances between all pairs of points
distances = {}
for i, p1 in enumerate(points):
    for j, p2 in enumerate(points):
        if i != j and (j, i) not in distances:
            distances[(i, j)] = calculate_distance(p1, p2)

# Plotting the points
plt.figure(figsize=(8, 8))
plt.scatter([point[0] for point in points], [point[1] for point in points], color='blue', label='Points')

# Annotate pairwise distances
for pair, distance in distances.items():
    point1, point2 = pair
    plt.plot([points[point1][0], points[point2][0]], [points[point1][1], points[point2][1]], linestyle='--', color='red')
    plt.text((points[point1][0] + points[point2][0]) / 2, (points[point1][1] + points[point2][1]) / 2, f'{distance:.2f}', color='black')

plt.xlabel('X-axis (cm)')
plt.ylabel('Y-axis (cm)')
plt.title('Randomly Generated Points within 100x100cm Box with Pairwise Distances')
plt.legend()
plt.grid(visible=True)
plt.show()

