import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread("/Users/hoangnamvu/Desktop/test.png", cv2.IMREAD_GRAYSCALE)

# Invert the image
inverted_image = 255 - image

# Display original and inverted images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(inverted_image, cmap="gray")
plt.title("Inverted Image")
plt.axis("off")

plt.show()

cv2.imwrite("output/test5.png", inverted_image)