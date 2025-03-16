import os
import cv2
import numpy as np

def process_images(folder_path, output_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    # Read the first image to determine dimensions
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first_image.shape
    accumulation_matrix = np.zeros((h, w), dtype=np.float64)

    # Accumulate pixel values
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (h, w):
            print(f"Skipping {img_file}: incompatible dimensions.")
            continue
        accumulation_matrix += img

    # Normalize the accumulated matrix to grayscale range (0-255)
    min_val, max_val = np.min(accumulation_matrix), np.max(accumulation_matrix)
    if max_val > 0:  # Avoid division by zero
        normalized_matrix = ((accumulation_matrix - min_val) / (max_val - min_val)) * 255
    else:
        normalized_matrix = accumulation_matrix  # All images were black
    
    normalized_matrix = normalized_matrix.astype(np.uint8)

    # Save the output image
    cv2.imwrite(output_path, normalized_matrix)
    print(f"Accumulated image saved at {output_path}")

# Example usage
folder_path = "data"  # Replace with your folder path
output_path = "output/test3.png"  # Output image path
process_images(folder_path, output_path)

