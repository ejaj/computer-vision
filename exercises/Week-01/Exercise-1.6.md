# Exercise 1.6

## Problem Statement

Given the matrix \( A \) and a point \( p \) in homogeneous coordinates:

\[
A = \begin{bmatrix}
10 & 0 & 2 \\
0 & 10 & -3 \\
0 & 0 & 1
\end{bmatrix}
\]

We want to find the result of \( Ap = q \), where \( p \) and \( q \) are 2D points in homogeneous coordinates.

## Solution

1. **Matrix Multiplication**:
   - Let's assume the point \( p \) in homogeneous coordinates is given by:
     \[
     p = \begin{bmatrix}
     x \\
     y \\
     1
     \end{bmatrix}
     \]
   - We multiply matrix \( A \) by point \( p \):
     \[
     q = A \cdot p = \begin{bmatrix}
     10 & 0 & 2 \\
     0 & 10 & -3 \\
     0 & 0 & 1
     \end{bmatrix}
     \begin{bmatrix}
     x \\
     y \\
     1
     \end{bmatrix}
     \]

2. **Performing the Multiplication**:
   - Compute each component of \( q \):
     \[
     q_x = 10x + 0y + 2 \cdot 1 = 10x + 2
     \]
     \[
     q_y = 0x + 10y - 3 \cdot 1 = 10y - 3
     \]
     \[
     q_w = 0x + 0y + 1 \cdot 1 = 1
     \]

   - So, the resulting point \( q \) in homogeneous coordinates is:
     \[
     q = \begin{bmatrix}
     10x + 2 \\
     10y - 3 \\
     1
     \end{bmatrix}
     \]

## Explanation of the Coefficients in \( A \)

- **Scaling Factors (10 and 10)**:
  - The \( 10 \)s on the diagonal of the matrix \( A \) (first two entries) scale the \( x \)- and \( y \)-coordinates by 10. This means each coordinate is multiplied by 10, enlarging the point by a factor of 10.

- **Translation Factors (2 and -3)**:
  - The entries \( 2 \) and \(-3 \) are responsible for translating the point. Specifically, \( 2 \) adds 2 to the \( x \)-coordinate, and \(-3 \) subtracts 3 from the \( y \)-coordinate. These translations happen after the scaling due to matrix multiplication order.

- **Last Diagonal Entry (1)**:
  - The last diagonal entry is \( 1 \), which means it doesn't change the homogeneous coordinate. If it were a different number (like 10), it would scale the entire homogeneous coordinate system but not affect the final inhomogeneous coordinates after division by the homogeneous scale factor.

## Conclusion

The direct solution to \( Ap = q \) is:

\[
q = \begin{bmatrix}
10x + 2 \\
10y - 3 \\
1
\end{bmatrix}
\]

The coefficients in \( A \) affect the coordinates of \( q \) as follows:
- The two 10’s scale the \( x \)- and \( y \)-coordinates.
- The 2 and -3 are translations in the \( x \)- and \( y \)-directions, respectively.
- The translations happen after scaling, ensuring the point is first enlarged and then shifted.

This explanation provides a clear understanding of how each component in the matrix \( A \) transforms the point \( p \) to \( q \) in homogeneous coordinates.
