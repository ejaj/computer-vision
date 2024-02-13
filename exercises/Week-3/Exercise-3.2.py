import numpy as np


def CrossOp(p):
    """
    Takes a vector p in 3D and returns the corresponding 3x3 skew-symmetric matrix
    for the cross product operation.
    """
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])


# Test the CrossOp function with random vectors
p1 = np.random.rand(3)
p2 = np.random.rand(3)

# Calculate the cross product using the CrossOp function
cross_op_result = CrossOp(p1) @ p2

# Calculate the cross product directly
direct_cross_product = np.cross(p1, p2)

# Check if the results are the same
cross_op_result, direct_cross_product, np.allclose(cross_op_result, direct_cross_product)
print(cross_op_result)
print(direct_cross_product)