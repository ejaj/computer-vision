# Homogeneous Coordinates for a 2D Line

To express the 2D line \( x + 2y = 3 \) in homogeneous coordinates, we need to rewrite it in the form \( l^T p = 0 \), where \( l \) is the homogeneous line representation and \( p \) is the homogeneous point.

## Step-by-Step Process:

1. **Rewrite the line equation**:
   The given line equation is \( x + 2y = 3 \). This can be rearranged as:
   \[
   x + 2y - 3 = 0
   \]

2. **Identify the coefficients**:
   The coefficients \( a \), \( b \), and \( c \) from the equation \( ax + by + c = 0 \) are:
   \[
   a = 1, \quad b = 2, \quad c = -3
   \]

3. **Form the homogeneous line representation**:
   The line in homogeneous form is represented by the vector \( l = [a, b, c]^T \):
   \[
   l = \begin{bmatrix}
   1 \\
   2 \\
   -3
   \end{bmatrix}
   \]

4. **Homogeneous point representation**:
   A point \( p \) in homogeneous coordinates is represented as \([x, y, w]^T \), where \( w \) is typically 1 for regular points.

## Homogeneous Form

The line equation in homogeneous form is:
\[
l^T p = 0
\]
where \( l = \begin{bmatrix} 1 \\ 2 \\ -3 \end{bmatrix} \) and \( p = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \).

So the homogeneous form of the line \( x + 2y = 3 \) is:
\[
\begin{bmatrix}
1 & 2 & -3
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix} = 0
\]

## Conclusion

The given line \( x + 2y = 3 \) in homogeneous coordinates is represented as:
\[
l^T p = 0
\]
where \( l = \begin{bmatrix} 1 \\ 2 \\ -3 \end{bmatrix} \) and \( p = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \).
