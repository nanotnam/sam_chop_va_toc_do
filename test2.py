import cv2
import numpy as np
import os

# Parameters
IMAGE_FOLDER = "data"  # Replace with your folder path

# Load images
image_files = sorted([os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))])
if not image_files:
    print("Error: No images found in the folder")
    exit()

# Read the first image to get dimensions
sample_image = cv2.imread(image_files[0])
height, width, _ = sample_image.shape
accumulation_map = np.zeros((height, width), dtype=np.uint16)

for img_path in image_files:
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    
    # Draw detected lines on an empty mask
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Accumulate detected lines
    accumulation_map += (line_mask > 0).astype(np.uint16)

# Normalize and threshold accumulation map
accumulation_map = (accumulation_map / accumulation_map.max() * 255).astype(np.uint8)
thresh_value = np.percentile(accumulation_map[accumulation_map > 0], 80)  # Keep top 20%
final_mask = (accumulation_map >= thresh_value).astype(np.uint8) * 255

# Post-processing to refine markings
kernel = np.ones((3, 3), np.uint8)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)


# Save results
cv2.imwrite("output/test2_1.png", accumulation_map)
cv2.imwrite("output/test2_2.png", final_mask)
