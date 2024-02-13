from sympy import Matrix, sqrt, Abs

# Define the line in homogeneous coordinates
l = Matrix([1/sqrt(2), 1/sqrt(2), -1])

# Define the points in homogeneous coordinates
p1 = Matrix([0, 0, 1])
p2 = Matrix([sqrt(2), sqrt(2)/2, 1])
p3 = Matrix([sqrt(2), sqrt(2)/4, 1])

# Function to calculate the shortest distance from a line to a point
def shortest_distance_line_point(line, point):
    # Numerator: the absolute value of the dot product of line and point
    numerator = Abs(line.dot(point))
    # Denominator: product of the absolute value of the homogeneous coordinate of the point
    # and the square root of the sum of the squares of the x and y components of the line
    denominator = Abs(point[2]) * sqrt(line[0]**2 + line[1]**2)
    # The shortest distance is the numerator divided by the denominator
    return numerator / denominator

# Calculate the shortest distances for each point
d1 = shortest_distance_line_point(l, p1).evalf()
d2 = shortest_distance_line_point(l, p2).evalf()
d3 = shortest_distance_line_point(l, p3).evalf()

# Print the results
print(f"The shortest distance from l to p1 is: {d1}")
print(f"The shortest distance from l to p2 is: {d2}")
print(f"The shortest distance from l to p3 is: {d3}")
