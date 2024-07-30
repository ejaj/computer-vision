import matplotlib.pyplot as plt
import numpy as np
import random


def draw_two_points(points):
    return random.sample(points, 2)


def calculate_line_params(point1, point2):
    if (point1[0] - point2[0]) == 0:
        m = float('inf')  # Infinite slope, vertical line
        c = point1[0]  # x-intercept since line is vertical
    else:
        m = (point1[1] - point2[1]) / (point1[0] - point2[0])
        c = point1[1] - m * point1[0]
    return m, c


def point_line_distance(point, m, c):
    if m == float('inf'):
        return abs(point[0] - c)
    return abs(m * point[0] - point[1] + c) / np.sqrt(m ** 2 + 1)


def calculate_consensus(points, m, c, threshold):
    inlier_count = 0
    for point in points:
        if point_line_distance(point, m, c) <= threshold:
            inlier_count += 1
    return inlier_count


# RANSAC Algorithm

def ransac_line_fitting(points, num_iterations, threshold):
    best_m = 0
    best_c = 0
    max_inliers = 0

    for _ in range(num_iterations):
        point1, point2 = draw_two_points(points)
        m, c = calculate_line_params(point1, point2)
        inliers = calculate_consensus(points, m, c, threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_m = m
            best_c = c

    return best_m, best_c, max_inliers


# Generate synthetic data (with a clear linear trend + noise)
np.random.seed(0)
x = np.random.uniform(-10, 10, 100)
y = 3 * x + 4 + np.random.normal(0, 3, 100)  # Line y = 3x + 4 with noise
points = list(zip(x, y))

# Parameters
num_iterations = 100
thresholds = np.linspace(0.5, 10, 20)  # Range of thresholds from 0.5 to 10

# Store results
results = []

for threshold in thresholds:
    m, c, inliers = ransac_line_fitting(points, num_iterations, threshold)
    results.append((threshold, inliers, m, c))

# Plotting results
thresholds, inliers, ms, cs = zip(*results)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(thresholds, inliers, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Number of Inliers')
plt.title('Threshold vs. Inliers')

plt.subplot(1, 2, 2)
plt.plot(thresholds, ms, marker='o', label='Slope')
plt.plot(thresholds, cs, marker='x', label='Intercept')
plt.xlabel('Threshold')
plt.legend()
plt.title('Line Parameters vs. Threshold')

plt.tight_layout()
plt.show()
