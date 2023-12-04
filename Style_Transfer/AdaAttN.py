# https://replicate.com/huage001/adaattn/api?tab=python

import replicate
import requests
import os

output = replicate.run(
    "huage001/adaattn:957250892e7125f4834c5b5e5b5b2b43dc18ff174a6d70958574d08298567a21",
    input={
        "content": open("style_3/house.jpg", "rb"),
        "style": open("model_inputs/style_1.jpg", "rb")
    }
)
# Access the model's output URI
model_output_uri = output

# Define the local file path where you want to save the output
local_output_path = "style_2/house_adaattn.jpg"

# Download the model's output and save it locally
response = requests.get(model_output_uri)

if response.status_code == 200:
    with open(local_output_path, 'wb') as output_file:
        output_file.write(response.content)
    print("Model output saved to:", local_output_path)
else:
    print("Failed to download the model's output.")

