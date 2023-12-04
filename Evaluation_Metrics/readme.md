# Evaluation Metrics Folder Overview

This repository includes two evaluation metrics designed for text-to-image models. The following provides a brief overview of these metrics:

1. **PSNR (Peak Signal-to-Noise Ratio)**
2. **Style Transfer**

## Instructions on How to Run the Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

To utilize the PSNR metric, follow the steps below:

1. Update line 37 with the path to the style image.
2. Update line 40 with the path to the model-generated image.
3. Run the following Python script:
   ```python
   python psnr.py
    ```

### Style Loss

For the Style Transfer metric, adhere to the following instructions:

1. Update line 10 with the path to the style image.
2. Update line 11 with the path to the model-generated image.
3. Run the following Python script:
    ```python
    python style_loss.py
    ```
