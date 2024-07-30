import numpy as np
import matplotlib.pyplot as plt


def test_points(n_in, n_out):
    a = (np.random.rand(n_in) - .5) * 10
    b = np.vstack((a, a * .5 + np.random.randn(n_in) * .25))
    points = np.hstack((b, 2 * np.random.randn(2, n_out)))
    return np.random.permutation(points.T).T


def data_plot(points):
    # Visualize initial data
    plt.figure(figsize=(10, 6))
    plt.scatter(points[0, :], points[1, :], color='blue', label='Data Points')
    plt.title('Generated Data Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Generate data
    n_inliers = 100
    n_outliers = 50
    points = test_points(n_inliers, n_outliers)
    data_plot(points)
