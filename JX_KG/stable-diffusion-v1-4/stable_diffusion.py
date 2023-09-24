# https://huggingface.co/CompVis/stable-diffusion-v1-4

import torch
from diffusers import StableDiffusionPipeline

########################### Running the pipeline with the default PNDM scheduler: ############################
model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id) 
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")