# Small script to test the API without using the browser
import sys
import requests

# take the image name from the command line
image_path = sys.argv[1]

# send the image to the /predict route
with open(image_path, "rb") as f:
    response = requests.post("http://127.0.0.1:8000/predict", files={"file": f})

# print what the server replied
print(response.json())
