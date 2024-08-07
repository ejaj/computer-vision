from PIL import Image
import numpy as np


def transformIm(im, theta, s):
    width, height = im.size
    new_width = int(width * s)
    new_height = int(height * s)
    scaled_im = im.resize((new_width, new_height), Image.LANCZOS)
    r_im = scaled_im.rotate(theta, expand=True)
    return r_im


im = Image.open('data/sunflowers.jpg')

theta = 45  #
s = 0.5
# Apply the transformation
r_im = transformIm(im, theta, s)

# Show the original and transformed images
im.show(title="Original Image")
r_im.show(title="Transformed Image")

# Save the transformed image if needed
r_im.save('transformed_example.jpg')
