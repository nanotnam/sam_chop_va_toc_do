import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread("output/test5.png", cv2.IMREAD_GRAYSCALE)

# Compute histogram along x-axis (sum pixel intensities along y-axis)
hist_x = np.sum(image, axis=0)

# Plot histogram
plt.figure(figsize=(10, 7))
plt.plot(hist_x, color='black')
plt.xlabel("X-axis (Columns)")
plt.ylabel("Sum of Pixel Intensities")
plt.title("Histogram Along X-Axis")
plt.show()
