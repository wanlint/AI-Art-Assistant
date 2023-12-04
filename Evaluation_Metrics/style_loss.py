import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image
import matplotlib.pyplot as plt

# Specify the file paths for the style and content images
style_image_path = 'model_inputs/style_1.jpg' 
content_image_path = 'ada.JPG' 

# Load style and content images using Pillow
style_image = Image.open(style_image_path)
content_image = Image.open(content_image_path)

# Preprocess the images (resize and normalize)
preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
style_image = preprocess(style_image).unsqueeze(0)  # Add a batch dimension
content_image = preprocess(content_image).unsqueeze(0)  # Add a batch dimension

# Load a pre-trained VGG model
vgg_model = models.vgg19(pretrained=True).features
vgg_model.eval()

# Define the style layers from which to extract feature maps
style_layers = [ 'conv_4', 'conv_5'] # Typically, deeper layers (e.g., conv_4 and conv_5) capture higher-level features that contribute more to the overall style of an image, while shallower layers (e.g., conv_1, conv_2) capture finer texture details.

# Extract feature maps from style and generated images
style_features = {}
generated_features = {}
for layer in style_layers:
    style_features[layer] = vgg_model(style_image)
    generated_features[layer] = vgg_model(content_image)

# Function to calculate the Gram matrix
def gram_matrix(input_tensor):
    batch, channels, height, width = input_tensor.size()
    features = input_tensor.view(channels, height * width)
    gram = torch.mm(features, features.t())
    return gram

# Calculate Gram matrices for style features and generated features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}
generated_grams = {layer: gram_matrix(generated_features[layer]) for layer in style_layers}

# Calculate style loss
style_loss = 0
for layer in style_layers:
    style_loss += torch.mean((generated_grams[layer] - style_grams[layer]) ** 2)

# Total loss (you may need to add content loss and other terms)
total_loss = style_loss
print("style_loss: ", total_loss)