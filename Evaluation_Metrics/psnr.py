
import cv2
import os
import numpy as np

MAX = 255  # Assuming 8-bit images

def calculate_psnr(image1, image2):
    # Load images using OpenCV
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    if image1 is None or image2 is None:
        raise ValueError("Error loading images")

    # Ensure that both images have the same dimensions
    if image1.shape[:2] != image2.shape[:2]:
        # Resize the images to have the same dimensions
        height, width = image1.shape[:2]
        image2 = cv2.resize(image2, (width, height))

    # Convert images to float32
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calculate the squared difference
    diff = (image1 - image2) ** 2

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean(diff)

    # Calculate PSNR
    psnr = 20 * np.log10(MAX) - 10 * np.log10(mse)

    return psnr

style_image = os.path.join("model_inputs", "dog_original.jpg")

style_folder = "style_2"
print("Psnr value for adaattn:", calculate_psnr(style_image, os.path.join(style_folder, "dog_adaattn.jpg")))
print("Psnr value for neural:", calculate_psnr(style_image, os.path.join(style_folder, "dog_neural.png")))