import random
import numpy as np


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


# Example Usage

points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(100)]
num_iterations = 100
threshold = 1.0

m, c, inliers = ransac_line_fitting(points, num_iterations, threshold)
print(f"Best line parameters: y = {m}x + {c}")
print(f"Number of inliers: {inliers}")
