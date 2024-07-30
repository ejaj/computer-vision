import numpy as np
import random


# Helper Functions

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


def pca_line(x):
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l @ x.mean(1)))
    return l


def ransac_line_fitting(points, num_iterations, threshold):
    best_m = 0
    best_c = 0
    max_inliers = 0
    best_inliers = []

    for _ in range(num_iterations):
        point1, point2 = draw_two_points(points)
        m, c = calculate_line_params(point1, point2)
        current_inliers = [p for p in points if point_line_distance(p, m, c) <= threshold]

        if len(current_inliers) > max_inliers:
            max_inliers = len(current_inliers)
            best_m = m
            best_c = c
            best_inliers = current_inliers

    if best_inliers:
        # Fit line to all inliers using PCA
        x = np.array(best_inliers).T
        line_params = pca_line(x)
        return line_params, max_inliers
    else:
        return (best_m, best_c, -best_m * np.mean([p[0] for p in points]) - np.mean([p[1] for p in points])), 0


# Example Usage

np.random.seed(0)
points = [(np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(100)]
num_iterations = 100
threshold = 1.0

line_params, inliers = ransac_line_fitting(points, num_iterations, threshold)
print(f"Line parameters (a, b, c): {line_params}")
print(f"Number of inliers: {inliers}")
