import cv2
import numpy as np
import os

# Parameters
IMAGE_FOLDER = "data"  # Replace with your folder path
WHITE_LOWER = np.array([0, 0, 200], dtype=np.uint8)
WHITE_UPPER = np.array([180, 50, 255], dtype=np.uint8)
YELLOW_LOWER = np.array([15, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# Load images
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("Error: No images found in the folder")
    exit()

# Read the first image to get dimensions
sample_image = cv2.imread(image_files[0])
height, width, _ = sample_image.shape
accumulation_map = np.zeros((height, width), dtype=np.uint16)
i = 0
for img_path in image_files:
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detect white and yellow pixels
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Accumulate detected pixels
    accumulation_map += (combined_mask > 0).astype(np.uint16)


# Normalize and threshold accumulation map
accumulation_map = (accumulation_map / accumulation_map.max() * 255).astype(np.uint8)
thresh_value = np.percentile(accumulation_map[accumulation_map > 0], 90)  # Keep top 20%
final_mask = (accumulation_map >= thresh_value).astype(np.uint8) * 255

# Post-processing (optional)
# kernel = np.ones((3, 3), np.uint8)
# final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)



# Save results
cv2.imwrite("output/test1_1.png", accumulation_map)
cv2.imwrite("output/test1_2.png", final_mask)
