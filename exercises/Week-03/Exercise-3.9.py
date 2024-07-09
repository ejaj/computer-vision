import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/TwoImageData.npy', allow_pickle=True).item()


# Compute the cross product matrix for the translation vector
def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


img1 = data['im1']
img2 = data['im2']
R1 = data['R1']
R2 = data['R2']
t1 = data['t1']
t2 = data['t2']
K = data['K']

# Compute relative rotation and translation from camera 1 to camera 2
R = R2 @ R1.T  # Rotation from camera 1 to camera 2
t = np.array(t2 - t1).reshape(-1)  # Translation from camera 1 to camera 2

# Compute the essential matrix E
E = skew_symmetric(t) @ R
F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)


# Function to draw epipolar line
def draw_line(l, shape):
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2] / q[2]
        if all(q >= 0) and all(q + 1 <= shape[1::-1]):
            return q

    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1 - shape[1]], [0, 1, 1 - shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    plt.plot(*np.array(P).T, color='yellow')  # Drawing in yellow for visibility


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Displaying the images
axs[0].imshow(img1, cmap='gray')
axs[0].set_title('Image 1')
axs[1].imshow(img2, cmap='gray')
axs[1].set_title('Image 2')


def onclick(event):
    if event.inaxes == axs[0]:  # Check if click is within the first image
        pt1 = np.array([event.xdata, event.ydata, 1])  # Homogeneous coordinates
        l = np.dot(F, pt1)  # Computing the epipolar line for image 2
        axs[1].clear()  # Clear previous lines/image
        axs[1].imshow(img2, cmap='gray')  # Re-display image 2
        draw_line(l, img2.shape)  # Draw the new epipolar line
        fig.canvas.draw()  # Refresh the plot


# Connecting the click event handler
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
