import numpy as np


def point_line_distance(point, m, c):
    """
    Calculate the perpendicular distance from a point to a line given by y = mx + c.
    """
    x0, y0 = point
    return abs(m * x0 - y0 + c) / np.sqrt(m ** 2 + 1)


def calculate_consensus(points, m, c, threshold):
    """
    Calculate the consensus (number of inliers) for a line y = mx + c with respect to a set of points.

    :param points: List of tuples (x, y) representing the points.
    :param m: Slope of the line.
    :param c: y-intercept of the line.
    :param threshold: Distance threshold to determine inliers.
    :return: Number of inliers, i.e., points within the given threshold distance to the line.
    """
    inlier_count = 0
    for point in points:
        if point_line_distance(point, m, c) <= threshold:
            inlier_count += 1
    return inlier_count


points = [(0, 0), (1, 5), (2, 5), (3, 7)]
m = 2
c = 1
threshold = 1.5

consensus = calculate_consensus(points, m, c, threshold)
print("Consensus (Number of Inliers):", consensus)
