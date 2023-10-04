# https://huggingface.co/CompVis/stable-diffusion-v1-4

import torch
from diffusers import StableDiffusionPipeline

########################### Running the pipeline with the default PNDM scheduler: ############################
model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id) 
pipe = pipe.to(device)

prompt = "a monkey eating a hamburger, where the mood of the paining is anxiety, using Surrealism and in Johannes Vermeer style"
image = pipe(prompt).images[0]  
    
image.save("starbucsk.png")