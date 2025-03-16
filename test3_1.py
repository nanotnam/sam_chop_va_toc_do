import cv2
import numpy as np

def apply_threshold(input_path, output_path, threshold=128):
    # Load the accumulated image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply threshold: keep pixels above threshold, set others to 0
    _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # Post-processing to refine markings
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)
    
    # Save the thresholded image
    cv2.imwrite(output_path, final_mask)
    print(f"Thresholded image saved at {output_path}")

# Example usage
input_image_path = "output/test3.png"  # Replace with the path of the accumulated image
output_image_path = "output/test3_2.png"  # Output path
apply_threshold(input_image_path, output_image_path, threshold=160)

