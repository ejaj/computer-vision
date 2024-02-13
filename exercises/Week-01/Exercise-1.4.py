import numpy as np

# Given line vector l from Exercise 1.3
l = np.array([1, 2, -3])

# Points given in homogeneous coordinates
p1 = np.array([3, 0, 1])
p2 = np.array([6, 0, 2])
p3 = np.array([1, 1, 1])
p4 = np.array([1, 1, 1])  # Note: p4 is given as the same as p3, this could be a typo in the exercise
p5 = np.array([-40, 110, 10])
p6 = np.array([1, 4, 1])

# Calculate l^T * p for each point to check if they are on the line
on_line_p1 = np.dot(l, p1)
on_line_p2 = np.dot(l, p2)
on_line_p3 = np.dot(l, p3)
on_line_p4 = np.dot(l, p4)
on_line_p5 = np.dot(l, p5)
on_line_p6 = np.dot(l, p6)

# Store results in a dictionary for easy interpretation
results_exercise_1_4 = {
    'p1': on_line_p1,
    'p2': on_line_p2,
    'p3': on_line_p3,
    'p4': on_line_p4,
    'p5': on_line_p5,
    'p6': on_line_p6
}

print(results_exercise_1_4)
