import numpy as np


def point_line_distance(point, m, c):
    """
    Calculate the perpendicular distance from a point to a line given by y = mx + c.
    """
    x0, y0 = point
    return abs(m * x0 - y0 + c) / np.sqrt(m ** 2 + 1)


def classify_points(points, m, c, threshold):
    """
    Classify points as inliers or outliers based on their distance to the line y = mx + c.
    A point is an inlier if its distance to the line is less than or equal to the threshold.

    :param points: List of tuples (x, y) representing the points.
    :param m: Slope of the line.
    :param c: y-intercept of the line.
    :param threshold: Distance threshold to determine inliers.
    :return: A dictionary with two keys 'inliers' and 'outliers', each containing a list of points.
    """
    inliers = []
    outliers = []
    for point in points:
        distance = point_line_distance(point, m, c)
        if distance <= threshold:
            inliers.append(point)
        else:
            outliers.append(point)
    return {'inliers': inliers, 'outliers': outliers}


points = [(0, 0), (1, 5), (2, 5), (3, 7)]
m = 2
c = 1
threshold = 1.5

result = classify_points(points, m, c, threshold)
print("Inliers:", result['inliers'])
print("Outliers:", result['outliers'])
